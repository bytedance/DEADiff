"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import (
    CLIPEncoder,
    CLIPPreTrainedModel,
    _expand_mask,
)

class HybridCLIPEmbedder(nn.Module):
    def __init__(self, pretrained_model_name_or_path, subfolders=["",""], ctx_begin_pos=2):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolders[0]
        )
        self.text_encoder = HybridCLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolders[1]
        )
        self._CTX_BEGIN_POS = ctx_begin_pos
        self.device = 'cuda'
    
    def encode(self, text, ctx_embeddings):
        if ctx_embeddings is None:
            input_ids = self.tokenizer(
                text,
                padding="max_length",
                # padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(self.device)
            bs = input_ids.shape[0]
        else:
            input_ids = self.tokenizer.model_max_length
            bs = ctx_embeddings.shape[0]
        encoder_hidden_states = self.text_encoder(
            input_ids=input_ids,
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=[self._CTX_BEGIN_POS] * bs,
        )[0]
        return encoder_hidden_states
    
    def forward(self, text, ctx_embeddings):
        return self.encode(text, ctx_embeddings)

class HybridCLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = HybridCLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def forward(
        self,
        ctx_embeddings: torch.Tensor = None,
        ctx_begin_pos: list = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return self.text_model(
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=ctx_begin_pos,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class HybridCLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = HybridCLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ctx_embeddings: torch.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        if isinstance(input_ids, int):
            input_shape = (ctx_embeddings.shape[0], input_ids)
            input_ids = [49406, ] + [49407] * (input_ids - 1)
            input_ids = torch.tensor([input_ids] * ctx_embeddings.shape[0]).to(ctx_embeddings.device)
        else:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=ctx_begin_pos,
        )

        bsz, seq_len = input_shape
        # if ctx_embeddings is not None:
        #     seq_len += ctx_embeddings.size(1)
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states.dtype
        ).to(hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=input_ids.device),
            input_ids.to(torch.int).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


class HybridCLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(
        self,
        ctx_embeddings: torch.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if ctx_embeddings is None:
            ctx_len = 0
        else:
            ctx_len = ctx_embeddings.shape[1]

        # seq_length = (
        #     input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        # ) + ctx_len
        seq_length = (
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        
        if ctx_embeddings is not None:
            input_embeds_ctx = []
            bsz = inputs_embeds.shape[0]
            for i in range(bsz):
                cbp = ctx_begin_pos[i]
                w = 0 if (ctx_embeddings[i]==0).all() else 1
                if cbp==-1 or w==0:
                    cbp=int(torch.nonzero(input_ids[i]==49407)[0].cpu().item())
                cbp = min(cbp, inputs_embeds[0].shape[0]-ctx_embeddings.shape[1])
                # print(cbp, cbp+16, w)
                prefix = inputs_embeds[i, :cbp]
                # remove the special token embedding
                suffix = inputs_embeds[i, cbp:]
                # print(prefix.shape, suffix.shape, ctx_embeddings[i].shape)
                _ctx_embedding = ctx_embeddings[i]*w+(1-w)*suffix[:ctx_embeddings[i].shape[0]]
                input_embeds_ctx.append(
                    torch.cat([prefix, _ctx_embedding, suffix], dim=0)[:-ctx_embeddings[i].shape[0]]
                )

            inputs_embeds = torch.stack(input_embeds_ctx, dim=0)
        embeddings = inputs_embeds + position_embeddings

        return embeddings