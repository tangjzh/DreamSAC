from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

class ZeroPadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, vocab_size):
        super(ZeroPadLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.vocab_size = vocab_size

    def forward(self, x):
        x = self.fc(x)
        zero_pad = -torch.inf * torch.ones(x.size(0), x.size(1), self.vocab_size, device=x.device)
        output = torch.cat([zero_pad, x], dim=2)
        return output

class MultiModalHeadModel(nn.Module):
    def __init__(self, llm, context, segment_length, text_vocab_size,
                 model_type='llama', enable_kd=False, 
                 **kwargs):
        super().__init__()

        self.llm = llm
        self.context = context
        self.segment_length = segment_length
        self.model_type = model_type
        self.enable_kd = enable_kd
        self.text_vocab_size = text_vocab_size
        self.sep_token = llm.config.vocab_size - 2  # the last token is used for sep
        self.token_for_sdf = llm.config.vocab_size - 1

        if self.model_type == 'llama':
            self.llm.lm_head = ZeroPadLayer(llm.config.hidden_size, 
                                            llm.config.vocab_size - text_vocab_size,
                                            text_vocab_size)
        elif self.model_type == 'gpt2':
            self.llm.lm_head = ZeroPadLayer(llm.config.n_embd, 
                                            llm.config.vocab_size - text_vocab_size,
                                            text_vocab_size)
        else:
            raise ValueError(f"model_type {self.model_type} is not supported.")

    def get_input_embeddings(self, input_ids):
        # B, T = input_ids.shape
        # text_ids, vision_ids = input_ids.split([self.text_length, T - self.text_length], dim=1)
        # input_ids = torch.cat([text_ids, (torch.ones(B) * self.sep_token).unsqueeze(1).to(input_ids.device), vision_ids])
        if self.model_type == 'llama':
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        elif self.model_type == 'gpt2':
            inputs_embeds  = self.llm.wte(input_ids)
        else:
            raise ValueError(f"model_type {self.model_type} is not supported.")
        return inputs_embeds

    # def create_attention_mask(self, input_tokens: torch.Tensor, image_type: Optional[str] = None):
    #     B, T = input_tokens.shape
    #     T_text, T_pixel = self.text_length, T - self.text_length

    #     attn_mask = torch.ones((B, T_text + T_pixel, T_text + T_pixel), dtype=torch.float32)

    #     attn_mask[:, :T_text, :T_text] = 1

    #     if image_type == "multi-view":
    #         attn_mask[:, T_text:, T_text:] = 1
    #     elif image_type == "video":
    #         causal_mask = torch.tril(torch.ones((T_pixel, T_pixel), dtype=torch.float32))
    #         attn_mask[:, T_text:, T_text:] = causal_mask

    #     return attn_mask
    
    @torch.no_grad()
    def generate(
        self,
        inputs_token,
        text_length,
        attention_mask=None,
        do_sample=True,
        temperature=1.0,
        top_k=100,
        max_new_tokens=None,
        pad_token_id=50256,
        action: Optional[torch.FloatTensor] = None,
    ):
        device = inputs_token.device
        token_per_dyna = ((max_new_tokens + 1) // (self.segment_length - self.context)) - 1
        B, T = inputs_token.size()
        for i in range(self.segment_length - self.context):
            inputs_embeds = self.get_input_embeddings(inputs_token)
            predicted_token = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=pad_token_id,
                top_k=top_k,
                max_new_tokens=token_per_dyna,
                use_cache=True,
            )
            predicted_token = (torch.concat([predicted_token, (torch.ones(B) * self.token_for_sdf).unsqueeze(1).to(device)], dim=1)
                               .to(predicted_token.dtype))
            inputs_token = torch.concat([inputs_token, predicted_token], dim=1).to(torch.int64)
            attention_mask = torch.concat([attention_mask, torch.ones(predicted_token.shape).to(predicted_token.device)], dim=1).to(torch.int64)

        assert inputs_token.size(1) == T + max_new_tokens + 1
        return inputs_token[:, text_length+1:]

    # input: tokens
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        text_length = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        inputs_embeds = self.get_input_embeddings(input_ids)

        x = self.llm(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            # output_hidden_states=False,
        )

        if self.enable_kd:
            pass

        return x
