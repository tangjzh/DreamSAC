import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel, PretrainedConfig, AutoTokenizer,
    AutoConfig, AutoModelForCausalLM
)
from diffusers import AutoencoderKL
from typing import Optional, List, Union
from PIL import Image
import torch.nn.functional as F
from einops import rearrange

class MultiModalConfig(PretrainedConfig):
    """
    Configuration class for MultiModalModel.
    """
    model_type = "multi_modal_model"

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        vae_name: str = "stabilityai/sd-vae-ft-ema",
        transformer_name: str = "gpt2",
        noise_dim: int = 512,
        max_text_length: int = 77,
        max_image_views: int = 4,
        latent_dim: int = 768,
        text_tokens: int = 256,
        pixel_tokens: int = 1024,
        segment_length: int = 4096,
        stop_token_id: int = 50256,  # GPT-2's end-of-text token
        und_kd: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer_name = tokenizer_name
        self.vae_name = vae_name
        self.transformer_name = transformer_name
        self.noise_dim = noise_dim
        self.max_text_length = max_text_length
        self.max_image_views = max_image_views
        self.latent_dim = latent_dim
        self.stop_token_id = stop_token_id
        self.und_kd = und_kd
        self.text_tokens = text_tokens
        self.pixel_tokens = pixel_tokens
        self.segment_length = segment_length

class MultiModalModel(PreTrainedModel):
    """
    A multi-modal model that integrates text and image inputs using a VAE and Transformer architecture.
    Inherits from PreTrainedModel and implements a custom generate method for latent space generation.
    """
    config_class = MultiModalConfig

    def __init__(self, config: MultiModalConfig):
        super().__init__(config)
        self.cfg = config

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        # Initialize Camera Encoder
        # TODO: Add Camera Input

        # Initialize Transformer
        transformer_config = AutoConfig.from_pretrained(config.transformer_name, trust_remote_code=True)
        self.transformer = AutoModelForCausalLM.from_config(transformer_config, trust_remote_code=True)
        self.token_for_sdf = self.transformer.config.vocab_size - 1

        # Embedding layers
        self.image_proj = nn.Linear(self.vae.config.latent_channels, transformer_config.hidden_size)



        # Final projection to latent tokens
        self.output_proj = nn.Linear(transformer_config.hidden_size, self.vae.config.latent_channels)

        # Stop token embedding
        self.stop_token_id = config.stop_token_id
        self.register_buffer("stop_token_embedding", torch.zeros(1, transformer_config.hidden_size))
        nn.init.normal_(self.stop_token_embedding, mean=0.0, std=transformer_config.initializer_range)

    def get_input_embeddings(self, input_ids):
        if self.cfg.transformer_name == 'llama':
            text_embeddings = self.transformer.get_input_embeddings()(input_ids)  # (batch_size, seq_len, hidden_size)
        elif self.cfg.transformer_name == 'gpt2':
            text_embeddings = self.transformer.wte(input_ids)
        else:
            raise ValueError(f"model_type {self.cfg.transformer_name} is not supported.")
        return text_embeddings
    
    def create_attention_mask(self, input_tokens: torch.Tensor, image_type: Optional[str] = None):
        B, _ = input_tokens.shape
        T_text, T_pixel = self.cfg.text_tokens, self.cfg.pixel_tokens

        attn_mask = torch.ones((B, T_text + T_pixel, T_text + T_pixel), dtype=torch.float32)

        attn_mask[:, :T_text, :T_text] = 1

        if image_type == "multi-view":
            attn_mask[:, T_text:, T_text:] = 1
        elif image_type == "video":
            causal_mask = torch.tril(torch.ones((T_pixel, T_pixel), dtype=torch.float32))
            attn_mask[:, T_text:, T_text:] = causal_mask

        return attn_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_type: Optional[str] = None,  # 'multi_view' or 'video'
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for training and inference.

        Args:
            input_ids (torch.Tensor, optional): Tokenized text input IDs (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Attention mask for text inputs (batch_size, seq_len).
            image_embeddings (torch.Tensor, optional): Image embeddings (batch_size, hidden_size).
            noise (torch.Tensor, optional): Noise tensor to be added (batch_size, noise_dim).
            image_type (str, optional): Type of image input ('multi_view' or 'video').
            labels (torch.Tensor, optional): Target latent tokens for supervised training.

        Returns:
            torch.Tensor: Generated latent image tokens or loss if labels are provided.
        """

        # Embed text
        inputs_embeds = self.get_input_embeddings(input_ids)

        # Attention mask for multi-view and video
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids, image_type)

        # TODO: Edit Positional Encoding

        x = self.transformer(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=self.cfg.und_kd, # understanding kd
        )

        if self.cfg.und_kd:
            pass

        return x
    
    @torch.no_grad()
    def generate(
        self,
        inputs_token,
        do_sample=True,
        temperature=1.0,
        top_k=100,
        max_new_tokens=None,
    ):
        device = inputs_token.device
        context_length = self.cfg.text_tokens + self.cfg.pixel_tokens
        token_per_dyna = ((max_new_tokens + 1) // (self.cfg.segment_length - context_length)) - 1
        B, T = inputs_token.size()
        for i in range(self.cfg.segment_length - context_length):
            inputs_embeds = self.get_input_embeddings(inputs_token)
            predicted_token = self.transformer.generate(
                inputs_embeds=inputs_embeds,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=token_per_dyna
            )
            inputs_token = torch.concat([inputs_token, predicted_token,
                                         (torch.ones(B) * self.token_for_sdf).unsqueeze(1).to(device)], dim=1).to(torch.int64)

        assert inputs_token.size(1) == T + max_new_tokens + 1  # +1 for the last token
        return inputs_token[:, :-1]  # the last token(sdf) is not used