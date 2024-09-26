from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.models.layers import DropPath
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

from transformers import PretrainedConfig

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import \
        flash_attn_varlen_qkvpacked_func
    has_flash_attn = True
except:
    print('FlashAttention2 is not installed.')
    has_flash_attn = False

class DreamVisionConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072,
        patch_size=16,
        image_size=32,
        sequence_length=10,  # Number of frames or views
        dropout=0.1,
        attention_dropout=0.1,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        initializer_factor=1.0,
        use_flash_attn=False,
        qkv_bias=True,
        norm_type='layer_norm',
        drop_path_rate=0.1,
        output_hidden_states=False,
        use_return_dict=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_factor = initializer_factor
        self.use_flash_attn = use_flash_attn
        self.qkv_bias = qkv_bias
        self.norm_type = norm_type
        self.drop_path_rate = drop_path_rate
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                          device=qkv.device)
                output = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                             indices, batch_size, seqlen),
                                   'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None

# Reuse existing DreamRMSNorm or FusedRMSNorm
class DreamRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

try:
    from apex.normalization import FusedRMSNorm

    DreamRMSNorm = FusedRMSNorm  # noqa

    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of DreamRMSNorm')
except ImportError:
    # using the normal DreamRMSNorm
    pass
except Exception:
    logger.warning('Discovered apex but it failed to load, falling back to DreamRMSNorm')
    pass

NORM2FN = {
    'rms_norm': DreamRMSNorm,
    'layer_norm': nn.LayerNorm,
}

class DreamAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DreamVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        if config.use_flash_attn and not has_flash_attn:
            print('Warning: Flash Attention is not available, use_flash_attn is set to False.')
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = DreamRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = DreamRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _naive_attn(self, x, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, N)
        
        if attention_mask is not None:
            # Assume attention_mask is additive mask with -inf for masked positions
            attn = attn + attention_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, attention_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        # FlashAttention expects attn_mask in a specific format, handle accordingly
        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, attention_mask=attention_mask, need_weights=need_weights, causal=False
        )
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-headed attention with optional causal mask.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, 1, seq_length)
                or (batch_size, 1, seq_length, seq_length). Typically contains 0 for allowed positions
                and -inf for masked positions.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, embed_dim)
        """
        if not self.use_flash_attn:
            return self._naive_attn(hidden_states, attention_mask=attention_mask)
        else:
            return self._flash_attn(hidden_states, attention_mask=attention_mask)


class DreamMLP(nn.Module):
    def __init__(self, config: DreamVisionConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class DreamVisionEmbeddings(nn.Module):
    """
    Embeddings for handling image sequences (multi-view or video).

    The input is expected to have shape (batch_size, sequence_length, channels, height, width).
    Each image in the sequence is embedded using patch embeddings, and temporal positional
    embeddings are added to account for the sequence order.
    """
    def __init__(self, config: DreamVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.sequence_length = config.sequence_length  # Number of frames or views

        # Patch embedding for individual images
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Class token for the entire sequence
        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # Positional embeddings for spatial patches
        num_patches_per_image = (self.image_size // self.patch_size) ** 2
        self.num_patches = num_patches_per_image * self.sequence_length
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_dim))

        # Optional temporal embeddings to encode the sequence position
        self.temporal_embedding = nn.Parameter(torch.randn(1, self.sequence_length, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.FloatTensor): Input tensor of shape 
                (batch_size, sequence_length, channels, height, width)

        Returns:
            torch.Tensor: Embedded representations of shape 
                (batch_size, 1 + sequence_length * num_patches, embed_dim)
        """
        batch_size, seq_length, channels, height, width = pixel_values.shape
        if seq_length != self.sequence_length:
            raise ValueError(f'Expected sequence_length={self.sequence_length}, but got {seq_length}')

        # Reshape to process all images in the batch at once
        pixel_values = pixel_values.view(batch_size * seq_length, channels, height, width)
        patch_embeds = self.patch_embedding(pixel_values)  # (batch_size * seq_length, embed_dim, H', W')
        H_p, W_p = patch_embeds.shape[-2], patch_embeds.shape[-1]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # (batch_size * seq_length, num_patches, embed_dim)

        # Reshape back to (batch_size, sequence_length, num_patches, embed_dim)
        patch_embeds = patch_embeds.view(batch_size, seq_length, -1, self.embed_dim)

        # Add temporal embeddings
        patch_embeds = patch_embeds + self.temporal_embedding.unsqueeze(2)  # Broadcasting over patches

        # Flatten the sequence and patch dimensions
        patch_embeds = patch_embeds.view(batch_size, -1, self.embed_dim)  # (batch_size, sequence_length * num_patches, embed_dim)

        # Add class token
        class_embeds = self.class_embedding.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # (batch_size, 1 + seq*num_patches, embed_dim)

        # Add positional embeddings
        embeddings = embeddings + self.position_embedding[:, :embeddings.size(1), :].to(embeddings.dtype)

        return embeddings

class DreamVisionEncoderLayer(nn.Module):
    """
    Single encoder layer for DreamVisionModel, identical to DreamVisionEncoderLayer.
    """
    def __init__(self, config: DreamVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = DreamAttention(config)
        self.mlp = DreamMLP(config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, 1, seq_length)
                or (batch_size, 1, seq_length, seq_length). Typically contains 0 for allowed positions
                and -inf for masked positions.

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        # Self-Attention Block
        normed_hidden = self.norm1(hidden_states)
        attn_output = self.attn(normed_hidden, attention_mask=attention_mask)
        hidden_states = hidden_states + self.drop_path1(attn_output * self.ls1)

        # MLP Block
        normed_hidden = self.norm2(hidden_states)
        mlp_output = self.mlp(normed_hidden)
        hidden_states = hidden_states + self.drop_path2(mlp_output * self.ls2)

        return hidden_states

class DreamVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of multiple DreamVisionEncoderLayer instances.
    """
    def __init__(self, config: DreamVisionConfig):
        super().__init__()
        self.config = config
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList([
            DreamVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass for the encoder.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, 1, 1, seq_length)
                or (batch_size, 1, seq_length, seq_length). Typically contains 0 for allowed positions
                and -inf for masked positions.
            output_hidden_states (bool, optional): Whether to return all hidden states.
            return_dict (bool, optional): Whether to return a ModelOutput object.

        Returns:
            Union[Tuple, BaseModelOutput]: Encoder outputs.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states,
                    attention_mask
                )
            else:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask
                )

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )

class DreamVisionModel(PreTrainedModel):
    """
    DreamVisionModel is a Vision Transformer (ViT) that accepts image sequences
    (multi-view or video) as input. It extends the DreamVisionModel to handle
    sequences of images by incorporating temporal or view-specific embeddings.
    """
    main_input_name = 'pixel_values'
    config_class = DreamVisionConfig
    _no_split_modules = ['DreamVisionEncoderLayer']

    def __init__(self, config: DreamVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = DreamVisionEmbeddings(config)
        self.encoder = DreamVisionEncoder(config)

        # Initialize weights and apply final processing
        self.init_weights()

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        """
        Resize positional embeddings when the image size changes.

        Args:
            old_size (int): Original image size.
            new_size (int): New image size.
            patch_size (int): Patch size.
        """
        pos_emb = self.embeddings.position_embedding
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, -1, self.embed_dim).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings
    
    def _generate_causal_mask(self, seq_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generate a causal mask for the given sequence length, excluding the class token.

        Args:
            seq_length (int): Length of the sequence (including class token).
            device (torch.device): Device to create the mask on.
            dtype (torch.dtype): Data type of the mask.

        Returns:
            torch.Tensor: Causal mask of shape (1, 1, seq_length, seq_length)
        """
        # seq_length includes class token
        # The class token (position 0) can attend to all tokens without masking
        # Tokens 1 to N can attend to the class token and previous tokens

        # Create lower triangular mask for tokens 1 to N
        mask_tokens = torch.tril(torch.ones((seq_length - 1, seq_length - 1), device=device, dtype=dtype))

        # Create mask for tokens 1 to N
        # They can attend to class token (position 0) and to previous tokens
        # So, prepend a column of zeros (allow attending to class token)
        mask_tokens = torch.cat([torch.zeros((seq_length - 1, 1), device=device, dtype=dtype), mask_tokens], dim=1)

        # Create mask for the class token (position 0): can attend to all tokens
        mask_cls = torch.zeros((1, seq_length), device=device, dtype=dtype)

        # Combine masks
        mask = torch.cat([mask_cls, mask_tokens], dim=0)  # (seq_length, seq_length)

        # Replace zeros with -inf and ones with 0.0
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, seq_length, seq_length)

        return mask

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            use_causal_mask: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Forward pass for DreamVisionModel.

        Args:
            pixel_values (torch.FloatTensor, optional): Input tensor of shape 
                (batch_size, sequence_length, channels, height, width)
            output_hidden_states (bool, optional): Whether to return all hidden states.
            return_dict (bool, optional): Whether to return a ModelOutput object.

        Returns:
            Union[Tuple, BaseModelOutputWithPooling]: Model outputs.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')

        if len(pixel_values.shape) != 5:
            raise ValueError(f'Expected pixel_values to have 5 dimensions (batch, seq, channels, height, width), but got {pixel_values.shape}')

        hidden_states = self.embeddings(pixel_values)  # (batch, 1 + seq*num_patches, embed_dim)
        
        # Determine if causal mask should be applied
        if use_causal_mask:
            seq_length = hidden_states.size(1)  # Includes class token
            # Typically, causal masks are not applied to the class token, so adjust accordingly
            # Here, we apply the causal mask to the entire sequence including the class token
            # Modify if you want to exclude the class token
            attention_mask = self._generate_causal_mask(seq_length, hidden_states.device, hidden_states.dtype)
        else:
            attention_mask = None
        
        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state  # (batch, seq_len, embed_dim)
        pooled_output = last_hidden_state[:, 0, :]  # (batch, embed_dim)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
