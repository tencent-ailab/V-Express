# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional

import torch
from einops import rearrange

from .attention import BasicTransformerBlock
from .attention import TemporalBasicTransformerBlock


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class ReferenceAttentionControl:
    def __init__(
            self,
            unet,
            mode="write",
            do_classifier_free_guidance=False,
            attention_auto_machine_weight=float("inf"),
            gn_auto_machine_weight=1.0,
            style_fidelity=1.0,
            reference_attn=True,
            reference_adain=False,
            fusion_blocks="midup",
            batch_size=1,
            reference_attention_weight=1.,
            audio_attention_weight=1.,
            reference_drop_rate=0.,
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.reference_attention_weight = reference_attention_weight
        self.audio_attention_weight = audio_attention_weight
        self.reference_drop_rate = reference_drop_rate
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size,
        )

    def register_reference_hooks(
            self,
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            dtype=torch.float16,
            batch_size=1,
            num_images_per_prompt=1,
            device=torch.device("cpu"),
            fusion_blocks="midup",
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        reference_drop_rate = self.reference_drop_rate
        reference_attention_weight = self.reference_attention_weight
        audio_attention_weight = self.audio_attention_weight
        dtype = dtype
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
                self,
                hidden_states: torch.FloatTensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                cross_attention_kwargs: Dict[str, Any] = None,
                class_labels: Optional[torch.LongTensor] = None,
                video_length=None,
        ):
            if self.use_ada_layer_norm:  # False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            # self.only_cross_attention = False
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states
                    if self.only_cross_attention
                    else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states
                        if self.only_cross_attention
                        else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )

                    if self.use_ada_layer_norm_zero:
                        attn_output = gate_msa.unsqueeze(1) * attn_output
                    hidden_states = attn_output + hidden_states

                    if self.attn2 is not None:
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )
                        self.bank.append(norm_hidden_states.clone())

                        # 2. Cross-Attention
                        attn_output = self.attn2(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=encoder_attention_mask,
                            **cross_attention_kwargs,
                        )
                        hidden_states = attn_output + hidden_states

                if MODE == "read":
                    hidden_states = (
                            self.attn1(
                                norm_hidden_states,
                                encoder_hidden_states=norm_hidden_states,
                                attention_mask=attention_mask,
                            )
                            + hidden_states
                    )

                    if self.use_ada_layer_norm:  # False
                        norm_hidden_states = self.norm1_5(hidden_states, timestep)
                    elif self.use_ada_layer_norm_zero:
                        (
                            norm_hidden_states,
                            gate_msa,
                            shift_mlp,
                            scale_mlp,
                            gate_mlp,
                        ) = self.norm1_5(
                            hidden_states,
                            timestep,
                            class_labels,
                            hidden_dtype=hidden_states.dtype,
                        )
                    else:
                        norm_hidden_states = self.norm1_5(hidden_states)

                    bank_fea = []
                    for d in self.bank:
                        if len(d.shape) == 3:
                            d = d.unsqueeze(1).repeat(1, video_length, 1, 1)
                        d = rearrange(d, "b t l c -> (b t) l c")

                        if reference_drop_rate != 0.:
                            drop_mask = torch.rand(d.shape[0]) < reference_drop_rate
                            d[drop_mask, ...] = 0.
                        bank_fea.append(d)

                    attn_hidden_states = self.attn1_5(
                        norm_hidden_states,
                        encoder_hidden_states=bank_fea[0],
                        attention_mask=attention_mask,
                    )

                    if reference_attention_weight != 1.:
                        attn_hidden_states *= reference_attention_weight

                    hidden_states = (attn_hidden_states + hidden_states)

                    # self.bank.clear()
                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )

                        attn_hidden_states = self.attn2(
                            norm_hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                        )

                        if audio_attention_weight != 1.:
                            attn_hidden_states *= audio_attention_weight

                        hidden_states = (attn_hidden_states + hidden_states)

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    # Temporal-Attention
                    if self.unet_use_temporal_attention:
                        d = hidden_states.shape[1]
                        hidden_states = rearrange(
                            hidden_states, "(b f) d c -> (b d) f c", f=video_length
                        )
                        norm_hidden_states = (
                            self.norm_temp(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm_temp(hidden_states)
                        )
                        hidden_states = (
                                self.attn_temp(norm_hidden_states) + hidden_states
                        )
                        hidden_states = rearrange(
                            hidden_states, "(b d) f c -> (b f) d c", d=d
                        )

                    return hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                            torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                       or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                       or isinstance(module, TemporalBasicTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )
                if isinstance(module, TemporalBasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, TemporalBasicTransformerBlock
                    )

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(
            self,
            writer,
            do_classifier_free_guidance=True,
            do_unconditional_forward=False,
            dtype=torch.float16,
    ):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks))
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (torch_dfs(writer.unet.mid_block) + torch_dfs(writer.unet.up_blocks))
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                if do_classifier_free_guidance:
                    r.bank = [torch.cat([torch.zeros_like(v), v]).to(dtype) for v in w.bank]
                elif do_unconditional_forward:
                    r.bank = [torch.zeros_like(v).to(dtype) for v in w.bank]
                else:
                    r.bank = [v.clone().to(dtype) for v in w.bank]

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                            torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                       or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                       or isinstance(module, TemporalBasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r in reader_attn_modules:
                r.bank.clear()
