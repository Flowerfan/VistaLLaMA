from typing import List, Optional, Tuple, Union
import os
import torch
import logging

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    # LlamaModel,
    # LlamaForCausalLM,
)
from .llama import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from .blip2 import BertLMHeadModel, BertConfig
from .utils import LayerNorm

DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"


class VisionConfig:
    def __init__(self):
        self.frame_size = 224
        self.patch_size = 14
        self.hidden_size = 1408
        self.use_vid_start_end = None
        self.vid_start_token = None
        self.vid_end_token = None
        self.vid_patch_token = None


class VistaLLaMAQformerConfig(LlamaConfig):
    model_type = "VistaLLaMAQformer"


class VistaLLaMALlamaModel(LlamaModel):
    config_class = VistaLLaMAQformerConfig

    def __init__(
        self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None
    ):  # TODO: Remove unused params
        super(VistaLLaMALlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_config = VisionConfig()

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            # self.mm_projector = nn.Linear(self.qformer_config.hidden_size, config.hidden_size)

    def initialize_vision_modules(
        self, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False
    ):
        vision_config = self.vision_config
        num_patches = (vision_config.frame_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(
                vision_config.hidden_size, self.config.hidden_size
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )
            self.mm_projector.load_state_dict(
                {k.split(".")[-1]: v for k, v in mm_projector_weights.items()}
            )

        return dict(video_token_len=num_patches, vision_config=vision_config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        orig_embeds_params = getattr(self, "orig_embeds_params", None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (
            input_ids.shape[1] != 1 or self.training
        ) and video_spatio_temporal_features is not None:
            video_features = self.mm_projector(video_spatio_temporal_features)
            dummy_video_features = torch.zeros(
                video_features.shape[1],
                1024,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            dummy_video_features = self.mm_projector(dummy_video_features)

            new_input_embeds = []
            cur_video_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.vid_patch_token).sum() == 0:
                    # Multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = (
                        cur_input_embeds + (0.0 * dummy_video_features).sum()
                    )
                    new_input_embeds.append(cur_input_embeds)
                    cur_video_idx += 1
                    continue
                if self.vision_config.use_vid_start_end:
                    if (cur_input_ids == self.vision_config.vid_start_token).sum() != (
                        cur_input_ids == self.vision_config.vid_end_token
                    ).sum():
                        raise ValueError(
                            "The number of video start tokens and video end tokens should be the same."
                        )
                    video_start_tokens = torch.where(
                        cur_input_ids == self.vision_config.vid_start_token
                    )[0]
                    for video_start_token_pos in video_start_tokens:
                        cur_video_features = video_features[cur_video_idx].to(
                            device=cur_input_embeds.device
                        )
                        num_patches = cur_video_features.shape[0]
                        # valid_patch = num_patches - (cur_video_features == 0).all(dim=1).sum()
                        valid_patch = (
                            num_patches
                            - (video_spatio_temporal_features[cur_video_idx] == 0)
                            .all(dim=1)
                            .sum()
                        )
                        if (
                            cur_input_ids[video_start_token_pos + num_patches + 1]
                            != self.vision_config.vid_end_token
                        ):
                            raise ValueError(
                                "The video end token should follow the video start token."
                            )
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:video_start_token_pos].detach(),
                                    cur_input_embeds[
                                        video_start_token_pos : video_start_token_pos
                                        + 1
                                    ],
                                    cur_video_features[:valid_patch],
                                    cur_input_embeds[
                                        video_start_token_pos
                                        + valid_patch
                                        + 1 : video_start_token_pos
                                        + num_patches
                                        + 2
                                    ],
                                    cur_input_embeds[
                                        video_start_token_pos + num_patches + 2 :
                                    ].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[: video_start_token_pos + 1],
                                    cur_video_features,
                                    cur_input_embeds[
                                        video_start_token_pos + num_patches + 1 :
                                    ],
                                ),
                                dim=0,
                            )
                        cur_video_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_video_features = video_features[cur_video_idx]
                    num_patches = cur_video_features.shape[0]
                    if (
                        cur_input_ids == self.vision_config.vid_patch_token
                    ).sum() != num_patches:
                        raise ValueError(
                            "The number of video patch tokens should be the same as the number of video patches."
                        )
                    masked_indices = torch.where(
                        cur_input_ids == self.vision_config.vid_patch_token
                    )[0]
                    mask_index_start = masked_indices[0]
                    if (
                        masked_indices
                        != torch.arange(
                            mask_index_start,
                            mask_index_start + num_patches,
                            device=masked_indices.device,
                            dtype=masked_indices.dtype,
                        )
                    ).any():
                        raise ValueError(
                            "The video patch tokens should be consecutive."
                        )
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start].detach(),
                                cur_video_features,
                                cur_input_embeds[
                                    mask_index_start + num_patches :
                                ].detach(),
                            ),
                            dim=0,
                        )
                    else:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start],
                                cur_video_features,
                                cur_input_embeds[mask_index_start + num_patches :],
                            ),
                            dim=0,
                        )
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_video_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(VistaLLaMALlamaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            visual_mask=visual_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class VistaLLaMAQformerLlamaForCausalLM(LlamaForCausalLM):
    config_class = VistaLLaMAQformerConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VistaLLaMALlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.initialize_qformer(
            self.model.config.num_qformer_token,
            self.model.vision_config.hidden_size,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def initialize_qformer(self, num_query_token, vision_width):
        self.num_query_token = num_query_token
        self.ln_vision = LayerNorm(vision_width)
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        self.qformer_config = encoder_config
        self.Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        self.query_tokens = query_tokens
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.vis_projector = nn.Linear(1024, vision_width)
        self.vlm_projector = nn.Linear(
            self.qformer_config.hidden_size, self.config.mm_hidden_size
        )

    def qformer_seq_forward(self, image_features, num_clips=4):
        update_features = []
        query_tokens_qa = self.query_tokens.expand(image_features.shape[0], -1, -1)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            if image_features.shape[-1] != 1408:
                image_features = self.vis_projector(image_features)
            image_features = self.ln_vision(image_features)
            feature_dim = image_features.ndim
            if feature_dim == 3:
                clip_features = image_features[:, :100].reshape(
                    image_features.shape[0], num_clips, -1, image_features.shape[-1]
                )
                for num in range(num_clips):
                    v_features = clip_features[:, num]
                    visual_features = self.Qformer.bert(
                        query_embeds=query_tokens_qa,
                        encoder_hidden_states=v_features,
                        return_dict=True,
                    )
                    visual_features = visual_features.last_hidden_state
                    update_features.append(visual_features)
                    query_tokens_qa = visual_features
            elif feature_dim == 4:
                interval = image_features.shape[1] // num_clips
                sidx = (image_features.shape[1] - 1) % interval
                for num in range(image_features.shape[1]):
                    # for num in range(num_clips):
                    # v_features = image_features[:, num * interval]
                    v_features = image_features[:, num]
                    visual_features = self.Qformer.bert(
                        query_embeds=query_tokens_qa,
                        encoder_hidden_states=v_features,
                        return_dict=True,
                    )
                    visual_features = visual_features.last_hidden_state[:, :self.num_query_token]
                    update_features.append(visual_features)
                    # query_tokens_qa = visual_features.detach()
                    query_tokens_qa = torch.cat([query_tokens_qa[:, :self.num_query_token], visual_features.detach()], dim=1)
                update_features = update_features[sidx::interval]
            update_features = torch.cat(update_features, dim=1)
            update_features = self.vlm_projector(update_features)
        return update_features

    def qformer_forward(self, image_features, num_clips=4):
        update_features = []
        query_tokens_qa = self.query_tokens.expand(image_features.shape[0], -1, -1)
        if image_features.shape[-1] != 1408:
            image_features = self.vis_projector(image_features)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            image_features = self.ln_vision(image_features)
            feature_dim = image_features.ndim
            if feature_dim == 3:
                visual_features = self.Qformer.bert(
                    query_embeds=query_tokens_qa,
                    encoder_hidden_states=image_features,
                    return_dict=True,
                )
                update_features = visual_features.last_hidden_state
            elif feature_dim == 4:
                interval = image_features.shape[1] // num_clips
                update_features = []
                # for i in range(num_clips):
                #     v_features = image_features[:, i*interval]
                for num in range(image_features.shape[1]):
                    v_features = image_features[:, num]
                    visual_features = self.Qformer.bert(
                        query_embeds=query_tokens_qa,
                        encoder_hidden_states=v_features,
                        return_dict=True,
                    ).last_hidden_state
                    update_features.append(visual_features)
                update_features = torch.cat(update_features, dim=1)
            update_features = self.vlm_projector(update_features)
        return update_features

    def load_from_pretrained(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint path is invalid")

        state_dict = checkpoint["model"]
        # print('state_dict',state_dict.keys())
        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % filename)

        return msg

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        B, T = video_spatio_temporal_features.shape[:2]
        image_features = video_spatio_temporal_features
        # image_features = video_spatio_temporal_features.flatten(0, 1)
        # query_tokens_qa = self.query_tokens.expand(image_features.shape[0], -1, -1)
        # visual_features = self.Qformer.bert(query_embeds=query_tokens_qa, encoder_hidden_states=image_features, return_dict=True)
        # visual_features = visual_features.last_hidden_state
        if self.config.qformer_type == 'normal':
            visual_features = self.qformer_forward(image_features)
        else:
            visual_features = self.qformer_seq_forward(image_features)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_mask=visual_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            video_spatio_temporal_features=visual_features,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # def _update_model_kwargs_for_generation(
    #     self,
    #     outputs,
    #     model_kwargs,
    #     **kwargs
    # ):
    #     # update past_key_values
    #     if "visual_mask" in model_kwargs:
    #         visual_mask = model_kwargs["visual_mask"]
    #         model_kwargs["visual_mask"] = torch.cat(
    #             [visual_mask, visual_mask.new_zeros((visual_mask.shape[0], 1))], dim=-1
    #         )
    #     return super()._update_model_kwargs_for_generation(
    #             outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder, **kwargs
    #         )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if self.config.use_visual_mask:
            visual_mask = input_ids.eq(self.model.vision_config.vid_patch_token)
        else:
            visual_mask = None
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "visual_mask": visual_mask,
                "video_spatio_temporal_features": kwargs.get(
                    "video_spatio_temporal_features", None
                ),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(
        self,
        mm_use_vid_start_end,
        tokenizer,
        device,
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    ):
        vision_config = self.get_model().vision_config
        vision_config.use_vid_start_end = mm_use_vid_start_end
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_vid_start_end:
            num_new_tokens = (
                tokenizer.add_tokens(
                    [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN],
                    special_tokens=True,
                )
                + 1
            )
            self.resize_token_embeddings(len(tokenizer))
            (
                vision_config.vid_start_token,
                vision_config.vid_end_token,
            ) = tokenizer.convert_tokens_to_ids(
                [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN]
            )

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)
                ]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VIDEO_PATCH_TOKEN]
        )[0]


AutoConfig.register("VistaLLaMAQformer", VistaLLaMAQformerConfig)
AutoModelForCausalLM.register(
    VistaLLaMAQformerConfig, VistaLLaMAQformerLlamaForCausalLM
)
