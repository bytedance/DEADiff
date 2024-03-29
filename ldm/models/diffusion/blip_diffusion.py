# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.
"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import random
import logging
import os

import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import tqdm
import kornia

from torch import nn
from transformers.activations import QuickGELUActivation as QuickGELU
from contextlib import nullcontext
from omegaconf import ListConfig

from lavis.common.utils import download_and_untar, is_url
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer

from ldm.util import log_txt_as_img, ismap, isimage, instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion


class ProjLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        super().__init__()

        # Dense1 -> Act -> Dense2 -> Drop -> Res -> Norm
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.act_fn = QuickGELU()
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_p)

        self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x):
        x_in = x

        x = self.LayerNorm(x)
        x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in

        return x


class DualBlipDiffusion(LatentDiffusion):
    def __init__(
        self,
        vit_model="clip_L",
        qformer_num_query_token=16,
        qformer_cross_attention_freq=1,
        qformer_pretrained_path=None,
        qformer_train=False,
        sd_train_text_encoder=False,
        proj_train=False,
        drop_subject_prob=0.15,
        subject_text_key='text',
        input_image_key='image',
        trainable_parameters=[],
        diffusers_pretrained_path=None,
        multi_dataloader_prob=None,
        cross_attention_train=False,
        *args, **kwargs
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(*args, **kwargs)

        self.multi_dataloader_prob = multi_dataloader_prob
        
        self.cond_stage_trainable = True # 强制在get_input取原始文本，不过CLIP text encoder
        self.num_query_token = qformer_num_query_token

        # BLIP-2
        self.style_blip = Blip2Qformer(
            vit_model=vit_model,
            num_query_token=qformer_num_query_token,
            cross_attention_freq=qformer_cross_attention_freq,
        )
        self.content_blip = Blip2Qformer(
            vit_model=vit_model,
            num_query_token=qformer_num_query_token,
            cross_attention_freq=qformer_cross_attention_freq,
        )
        if qformer_pretrained_path is not None:
            state_dict = torch.load(qformer_pretrained_path, map_location="cpu")[
                "model"
            ]
            # qformer keys: Qformer.bert.encoder.layer.1.attention.self.key.weight
            # ckpt keys: text_model.bert.encoder.layer.1.attention.self.key.weight
            for k in list(state_dict.keys()):
                if "text_model" in k:
                    state_dict[k.replace("text_model", "Qformer")] = state_dict.pop(k)

            msg = self.style_blip.load_state_dict(state_dict, strict=False)
            assert all(["visual" in k for k in msg.missing_keys])
            assert len(msg.unexpected_keys) == 0
            
            msg = self.content_blip.load_state_dict(state_dict, strict=False)
            assert all(["visual" in k for k in msg.missing_keys])
            assert len(msg.unexpected_keys) == 0

        self.qformer_train = qformer_train

        # projection layer
        self.style_proj_layer = ProjLayer(
            in_dim=768, out_dim=768, hidden_dim=3072, drop_p=0.1, eps=1e-12
        )
        self.content_proj_layer = ProjLayer(
            in_dim=768, out_dim=768, hidden_dim=3072, drop_p=0.1, eps=1e-12
        )
        self.proj_train = proj_train

        self.sd_train_text_encoder = sd_train_text_encoder
        self.cross_attention_train = cross_attention_train

        self.freeze_modules()

        self.ctx_embeddings_cache = nn.Parameter(
            torch.zeros(1, self.num_query_token, 768), requires_grad=False
        )
        self._use_embeddings_cache = False

        self.restarted_from_ckpt = False
        if diffusers_pretrained_path is not None:
            self.load_blip_checkpoint_from_dir(diffusers_pretrained_path)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.drop_subject_prob = drop_subject_prob
        self.subject_text_key = subject_text_key
        self.input_image_key = input_image_key
        self.trainable_parameters = trainable_parameters
        if len(self.trainable_parameters)==1 and self.trainable_parameters[0] == 'none':
            self.model.eval()
            self.model.train = self.disabled_train
            for param in self.model.parameters():
                param.requires_grad = False
        # inference-related
        self.register_buffer('clip_mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('clip_std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        count = 0
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
            if "attn2.to_k" in k:
                sd[k.replace('attn2.to_k', 'attn2.i_to_k')] = sd[k]
                print(k, k.replace('attn2.to_k', 'attn2.i_to_k'))
                count += 1
            if "attn2.to_v" in k:
                sd[k.replace('attn2.to_v', 'attn2.i_to_v')] = sd[k]
                print(k, k.replace('attn2.to_v', 'attn2.i_to_v'))
                count += 1

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    def freeze_modules(self):
        to_freeze = [self.first_stage_model]
        if not self.cross_attention_train:
            to_freeze.append(self.model)

        if not self.sd_train_text_encoder:
            to_freeze.append(self.cond_stage_model)

        if not self.qformer_train:
            to_freeze.append(self.style_blip)
            to_freeze.append(self.content_blip)

        if not self.proj_train:
            to_freeze.append(self.style_proj_layer)
            to_freeze.append(self.content_proj_layer)

        for module in to_freeze:
            module.eval()
            module.train = self.disabled_train
            module.requires_grad_(False)
    
    def configure_optimizers(self):
        lr = self.learning_rate
        if len(self.trainable_parameters)>0:
            params = []
            for param in self.model.named_parameters():
                if 'ema' in param[0]:
                    continue
                for k in self.trainable_parameters:
                    if k in param[0]:
                        params.append(param[1])
                        break
        else:
            params = list(self.model.parameters())
        if self.cross_attention_train:
            count = 0
            for name, param in self.model.named_parameters():
                if "attn2.i_to_k" in name or "attn2.i_to_v" in name:
                    print(name)
                    params.append(param)
                    count += 1
            print(count)
        if self.qformer_train:
            params += list(self.style_blip.Qformer.parameters())
            params += list(self.content_blip.Qformer.parameters())
            params += [self.style_blip.query_tokens]
            params += [self.content_blip.query_tokens]
        if self.proj_train:
            params += list(self.style_proj_layer.parameters())
            params += list(self.content_proj_layer.parameters())
        if self.sd_train_text_encoder:
            params += list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
    
    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    def get_learned_conditioning(self, c):
        if 'inp_image' in c:
            inp_image = c["inp_image"]
            inp_image = (inp_image+1)/2
            # print(inp_image.min(), inp_image.max())
            inp_image = kornia.geometry.resize(inp_image, (224, 224),
                                    interpolation='bicubic', align_corners=True,
                                    antialias=True)
            # re-normalize according to clip
            inp_image = kornia.enhance.normalize(inp_image, self.clip_mean, self.clip_std)
            if isinstance(c["subject_text"][0], list):
                style_subject_text = [row[0] for row in c["subject_text"]]
                style_ctx_embeddings = self.forward_ctx_embeddings(
                    input_image=inp_image, text_input=style_subject_text, blip=self.style_blip, proj_layer=self.style_proj_layer
                )
                if self.training:
                    text_encoder_hidden_states_style = self.cond_stage_model.encode(['' for i in c['target_text']])
                else:
                    text_encoder_hidden_states_style = self.cond_stage_model.encode(c['target_text'])
                content_subject_text = [row[1] for row in c["subject_text"]]
                content_ctx_embeddings = self.forward_ctx_embeddings(
                    input_image=inp_image, text_input=content_subject_text, blip=self.content_blip, proj_layer=self.content_proj_layer
                )
                if self.training:
                    text_encoder_hidden_states_content = self.cond_stage_model.encode(['' for i in c['target_text']])
                else:
                    text_encoder_hidden_states_content = self.cond_stage_model.encode(c['target_text'])
                if self.training:
                    rand_num = random.randint(0, 2)
                    if rand_num == 0:
                        encoder_hidden_states_style = style_ctx_embeddings*c.get('drop_mask', 1)
                        encoder_hidden_states_content = content_ctx_embeddings
                    elif rand_num == 1:
                        encoder_hidden_states_content = content_ctx_embeddings*c.get('drop_mask', 1)
                        encoder_hidden_states_style = style_ctx_embeddings
                    else:
                        encoder_hidden_states_style = style_ctx_embeddings*c.get('drop_mask', 1)
                        encoder_hidden_states_content = content_ctx_embeddings*c.get('drop_mask', 1)
                else:
                    encoder_hidden_states_style = style_ctx_embeddings
                    encoder_hidden_states_content = content_ctx_embeddings
            elif c["subject_text"][0] == 'style':
                ctx_embeddings = self.forward_ctx_embeddings(
                    input_image=inp_image, text_input=c["subject_text"], blip=self.style_blip, proj_layer=self.style_proj_layer
                )
                encoder_hidden_states_style = ctx_embeddings*c.get('drop_mask', 1)
                encoder_hidden_states_content = torch.zeros_like(encoder_hidden_states_style)
                text_encoder_hidden_states_style = text_encoder_hidden_states_content = self.cond_stage_model.encode(c['target_text'])
            else:
                ctx_embeddings = self.forward_ctx_embeddings(
                    input_image=inp_image, text_input=c["subject_text"], blip=self.content_blip, proj_layer=self.content_proj_layer
                )
                encoder_hidden_states_content = ctx_embeddings*c.get('drop_mask', 1)
                encoder_hidden_states_style = torch.zeros_like(encoder_hidden_states_content)
                text_encoder_hidden_states_content = text_encoder_hidden_states_style = self.cond_stage_model.encode(c['target_text'])
            encoder_hidden_states = [[encoder_hidden_states_style, text_encoder_hidden_states_style], [encoder_hidden_states_content, text_encoder_hidden_states_content]]
        else:
            text_encoder_hidden_states = self.cond_stage_model.encode(c['target_text'])
            encoder_hidden_states = [torch.zeros_like(text_encoder_hidden_states)[:, :self.num_query_token, :], text_encoder_hidden_states]
        return encoder_hidden_states
    
    def get_input(self, batch, k, cond_key=None, bs=None, **kwargs):
        batch_keys = list(batch.keys())
        if isinstance(batch_keys[0], int):
            batch_key = random.choices(batch_keys, weights=self.multi_dataloader_prob)[0]
            drop_subject_prob = self.drop_subject_prob[batch_key]
            batch = batch[batch_key]
        else:
            drop_subject_prob = self.drop_subject_prob[-1]
        outputs = super().get_input(batch, k, bs=bs, **kwargs)
        c = outputs[1] # str text
        inp_image = batch.get(self.input_image_key, batch.get(self.first_stage_key))[:bs]
        
        if self.subject_text_key=='first_tag':
            subject_text = c
            subject_text = [_c.split(',')[0] for _c in subject_text]
        elif self.subject_text_key.startswith('first_random:'):
            _, first_n, random_n = self.subject_text_key.split(':')
            first_n, random_n = int(first_n), int(random_n)
            subject_text = c
            subject_text = [','.join(_c.split(',')[:first_n]) for _c in subject_text]
        else:
            subject_text = batch[self.subject_text_key][:bs]
        c = {'target_text':c, 'inp_image':inp_image, 'subject_text':subject_text}
        if self.training:
            c['drop_mask'] = (torch.rand(len(inp_image)) >= drop_subject_prob).reshape(len(inp_image),1,1).to(inp_image.device)[:bs]
        else:
            c['drop_mask'] = (torch.rand(len(inp_image)) >= 0.0).reshape(len(inp_image),1,1).to(inp_image.device)[:bs]
        outputs[1] = c
        return outputs

    def forward_ctx_embeddings(self, input_image, text_input, blip, proj_layer, ratio=None):
        def compute_ctx_embeddings(input_image, text_input, blip, proj_layer):
            # blip_embeddings = self.blip(image=input_image, text=text_input)
            blip_embeddings = blip.extract_features(
                {"image": input_image, "text_input": text_input}, mode="multimodal"
            ).multimodal_embeds
            ctx_embeddings = proj_layer(blip_embeddings)

            return ctx_embeddings

        if isinstance(text_input, str):
            text_input = [text_input]

        if self._use_embeddings_cache:
            # expand to batch size
            ctx_embeddings = self.ctx_embeddings_cache.expand(len(text_input), -1, -1)
        else:
            if isinstance(text_input[0], str):
                text_input, input_image = [text_input], [input_image]

            all_ctx_embeddings = []

            for inp_image, inp_text in zip(input_image, text_input):
                ctx_embeddings = compute_ctx_embeddings(inp_image, inp_text, blip, proj_layer)
                all_ctx_embeddings.append(ctx_embeddings)

            if ratio is not None:
                assert len(ratio) == len(all_ctx_embeddings)
                assert sum(ratio) == 1
            else:
                ratio = [1 / len(all_ctx_embeddings)] * len(all_ctx_embeddings)

            ctx_embeddings = torch.zeros_like(all_ctx_embeddings[0])

            for ratio, ctx_embeddings_ in zip(ratio, all_ctx_embeddings):
                ctx_embeddings += ratio * ctx_embeddings_

        return ctx_embeddings


    def load_blip_checkpoint_from_dir(self, checkpoint_dir_or_url):
        # if checkpoint_dir is a url, download it and untar it
        if is_url(checkpoint_dir_or_url):
            checkpoint_dir_or_url = download_and_untar(checkpoint_dir_or_url)

        logging.info(f"Loading pretrained model from {checkpoint_dir_or_url}")

        def load_state_dict(module, filename):
            try:
                state_dict = torch.load(
                    os.path.join(checkpoint_dir_or_url, filename), map_location="cpu"
                )
                msg = module.load_state_dict(state_dict, strict=False)
            except FileNotFoundError:
                logging.info("File not found, skip loading: {}".format(filename))

        load_state_dict(self.style_proj_layer, "proj_layer/proj_weight.pt")
        load_state_dict(self.style_blip, "blip_model/blip_weight.pt")
        load_state_dict(self.content_proj_layer, "proj_layer/proj_weight.pt")
        load_state_dict(self.content_blip, "blip_model/blip_weight.pt")

        try:
            self.ctx_embeddings_cache.data = torch.load(
                os.path.join(
                    checkpoint_dir_or_url, "ctx_embeddings_cache/ctx_embeddings_cache.pt"
                ),
                map_location=self.device,
            )
            self._use_embeddings_cache = True
            print("Loaded ctx_embeddings_cache from {}".format(checkpoint_dir_or_url))
        except FileNotFoundError:
            self._use_embeddings_cache = False
            print("No ctx_embeddings_cache found in {}".format(checkpoint_dir_or_url))

    def apply_model(self, x_noisy, t, cond, img_weight=1.0, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, img_weight, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=50, ddim_eta=0., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        batch_keys = list(batch.keys())
        if isinstance(batch_keys[0], int):
            batch = batch[random.choices(batch_keys, weights=self.multi_dataloader_prob)[0]]
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        if self.use_bf16:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=self.use_bf16):
            z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                            return_first_stage_outputs=True,
                                            force_c_encode=False,
                                            return_original_cond=True,
                                            bs=N)
            N = min(x.shape[0], N)
            n_row = min(x.shape[0], n_row)
            log["inputs"] = x
            log["reconstruction"] = xrec
            if 'inp_image' in c:
                log["inp_image"] = c['inp_image']
            if self.model.conditioning_key is not None:
                if hasattr(self.cond_stage_model, "decode"):
                    xc = self.cond_stage_model.decode(c)
                    log["conditioning"] = xc
                elif self.cond_stage_key in ["caption", "text"]:
                    xc = log_txt_as_img((x.shape[3], x.shape[2]), batch[self.cond_stage_key], size=(x.shape[2]+x.shape[3]) // 64)
                    log["conditioning"] = xc
                elif self.cond_stage_key in ['class_label', "cls"]:
                    try:
                        xc = log_txt_as_img((x.shape[3], x.shape[2]), batch["human_label"], size=(x.shape[2]+x.shape[3]) // 64)
                        log['conditioning'] = xc
                    except KeyError:
                        # probably no "human_label" in batch
                        pass
                elif isimage(xc):
                    log["conditioning"] = xc
                if ismap(xc):
                    log["original_conditioning"] = self.to_rgb(xc)

            if unconditional_guidance_scale > 1.0:
                uc_N = N
                if isinstance(unconditional_guidance_label, ListConfig):
                    unconditional_guidance_label = list(unconditional_guidance_label)
                uc = self.get_learned_conditioning({'target_text':uc_N * unconditional_guidance_label})
                c = self.get_learned_conditioning(c)
                with ema_scope("Sampling with classifier-free guidance"):
                    samples_cfg, _ = self.sample_log(cond=c, batch_size=uc_N, ddim=use_ddim,
                                                    ddim_steps=ddim_steps, eta=ddim_eta,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=[uc, uc],
                                                    )
                    x_samples_cfg = self.decode_first_stage(samples_cfg)
                    log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            return log
