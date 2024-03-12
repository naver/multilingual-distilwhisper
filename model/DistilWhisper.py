import math
import random
import copy
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import datasets
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.hub import cached_file
#from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13 # TO DO - uncomment this for newer versions of Transformers
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
from transformers.models.whisper.modeling_whisper import WhisperAttention, WhisperPositionalEmbedding, WhisperEncoder, \
                                                        WhisperDecoder, WhisperEncoderLayer, WhisperDecoderLayer
#from transformers.models.whisper.modeling_whisper import _make_causal_mask, _expand_mask, _compute_mask_indices, shift_tokens_right
from old_hf_whisper import _make_causal_mask, _expand_mask, _compute_mask_indices, shift_tokens_right

from transformers import WhisperForConditionalGeneration, WhisperModel

logger = logging.get_logger('models')

DISTILWHISPER_ADAPTER_PT_FILE = "expert.{}.bin"
DISTILWHISPER_ADAPTER_SAFE_FILE = "expert.{}.safetensors"

def check_torch_parameters(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
                return False
            else:
                print('Mismtach found at key name')
                return False
    if models_differ == 0:
        return True

@dataclass
class CLSRSeq2SeqLMOutput(Seq2SeqLMOutput):
    gate_loss: Optional[torch.FloatTensor] = None
    encoder_gate_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_gate_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class CLSRSeq2SeqOutput(Seq2SeqModelOutput):
    encoder_gate_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_gate_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class CLSRModelOutput(BaseModelOutput):
    gate_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class CLSRModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    gate_states: Optional[Tuple[torch.FloatTensor]] = None


class DistilWhisperConfig(WhisperConfig):
    model_type = "distilwhisper"

    def __init__(
        self,
        clsr_langs: list[str] = None,
        skip_gate_prob=0.0, # ignore domain specific gate with certain probability (to allow for unseen domains to use shared representations only)
        clsr_gate_dim=64, # inner dimension of the feed-forward network used for computing the gate values
        clsr_max_steps=10000, # number of steps before the gate reachest its peakiest values (usually same as --max-steps)
        clsr_residual=False,
        use_gate_budget=True,
        gate_budget=1.0,
        **kwargs,
    ):
        self.clsr_langs = clsr_langs
        self.skip_gate_prob = skip_gate_prob
        self.clsr_gate_dim = clsr_gate_dim
        self.clsr_max_steps = clsr_max_steps
        self.clsr_residual = clsr_residual
        self.use_gate_budget = use_gate_budget
        self.gate_budget = gate_budget
        super().__init__(**kwargs)

    @classmethod
    def from_whisperconfig(
            cls,
            config,
            clsr_langs: list[str],
            skip_gate_prob=0.0,
            clsr_gate_dim=64,
            clsr_max_steps=10000,
            clsr_residual=False,
            use_gate_budget=True,
            gate_budget=1.0,
    ):
        kwargs = vars(config)
        return cls(clsr_langs=clsr_langs, skip_gate_prob=skip_gate_prob, clsr_gate_dim=clsr_gate_dim, clsr_max_steps=clsr_max_steps, clsr_residual=clsr_residual, use_gate_budget=use_gate_budget, gate_budget=gate_budget, **kwargs)

class FFN(nn.Module):
    def __init__(self, activation_function, activation_dropout, embed_dim, ffn_dim):
        super().__init__()
        self.activation_fn = ACT2FN[activation_function]
        self.activation_dropout = activation_dropout
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, hidden_states, activation_dropout, dropout, training):
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=activation_dropout, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=dropout, training=training)
        return hidden_states

    def set_from_pretrained(self, fc1, fc2, activation_fn):
        self.fc1 = copy.deepcopy(fc1)
        self.fc2 = copy.deepcopy(fc2)
        self.activation_fn = copy.deepcopy(activation_fn)

class CLSRFFN(nn.Module):
    def __init__(
        self,
        config: DistilWhisperConfig, # TO DO: correct
        uids: list[str],
        activation_function: str,
        activation_dropout: float,
        embed_dim: int,
        ffn_dim: int,
    ):
        super().__init__()
        uids = uids or []
        if "shared" not in uids:
            uids += ["shared"]
        self.skip_gate_prob = config.skip_gate_prob
        self.gate_in = nn.ModuleDict({uid: nn.Linear(embed_dim, config.clsr_gate_dim, bias=True) for uid in uids if uid != "shared"})
        self.gate_out = nn.ModuleDict({uid: nn.Linear(config.clsr_gate_dim, 1, bias=False) for uid in uids if uid != "shared"})
        self.proj = nn.ModuleDict({uid: FFN(activation_function, activation_dropout, embed_dim, ffn_dim) for uid in uids})
        self.max_steps = config.clsr_max_steps
        self.residual = config.clsr_residual
        for gate_in in self.gate_in.values():
            nn.init.normal_(gate_in.weight, mean=0, std=0.01)
            nn.init.zeros_(gate_in.bias)
        for gate_out in self.gate_out.values():
            nn.init.zeros_(gate_out.weight)
        for proj in self.proj.values():
            nn.init.normal_(proj.fc1.weight, mean=0, std=0.01)
            nn.init.zeros_(proj.fc1.bias)
            nn.init.zeros_(proj.fc2.weight)
            nn.init.zeros_(proj.fc2.bias)
    
    def forward(self, uid: str, step_proportion: float, input: torch.Tensor, activation_dropout, dropout, training, is_warm_up=False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Shape:
            input: (B, T, D)
        Returns: tuple (output, gate_value) with
            output: a tensor of shape (B, T, D)
            gate_value: a tensor of shape (B, T)
        """
        if not self.proj:
            return input
        if uid:
            G = self.gate_in[uid](nn.functional.dropout(input, p=0.1, training=training))
            G = F.relu(G)
            G = self.gate_out[uid](G)
            noise = torch.normal(0, 1, size=G.shape).to(G)
            if self.training and step_proportion:
                alpha = min(step_proportion, 1) * 5
            else:
                alpha = 5
            g = torch.sigmoid(G + alpha * noise)
            bsz, seq_len, dim = input.shape
            x = input.view(-1, dim)
            g = g.view(-1, 1)
            G = G.view(-1, 1)
            if not training:
                g = (G >= 0.0).to(g)   # hard gate at decoding
            skip_gate = False
            if training and self.skip_gate_prob > 0:
                # ignore domain specific gate with prob = cfg.skip_gate_prob
                p = random.uniform(0,1)
                if p < self.skip_gate_prob:
                    skip_gate = True
        if uid and is_warm_up:
            x = self.proj[uid](x, activation_dropout=activation_dropout, dropout=0.3, training=training)
        elif uid in self.proj and not skip_gate:
            h_shared = (1 - g) * self.proj['shared'](x, activation_dropout=activation_dropout, dropout=dropout, training=training)
            h_lang = g * self.proj[uid](x, activation_dropout=activation_dropout, dropout=0.3, training=training)
            x = h_lang + h_shared
        else:
            x = self.proj['shared'](x, activation_dropout=activation_dropout, dropout=dropout, training=training)
        
        if not uid:
            g = torch.zeros(g.shape).to(g)
        
        x = x.view(bsz, seq_len, dim)
        return x, g.view(bsz, seq_len)
    
    def _restart_gates(self):
        for gate_in in self.gate_in.values():
            nn.init.normal_(gate_in.weight, mean=0, std=0.01)
            nn.init.zeros_(gate_in.bias)
        for gate_out in self.gate_out.values():
            nn.init.zeros_(gate_out.weight)

    def add_language(self, lang, config):
        target_device = next(self.proj["shared"].parameters()).device
        self.proj[lang] = copy.deepcopy(self.proj["shared"]).to(target_device)
        self.gate_in[lang] = nn.Linear(config.d_model, config.clsr_gate_dim, bias=True).to(target_device)
        self.gate_out[lang] = nn.Linear(config.clsr_gate_dim, 1, bias=False).to(target_device)

class DistilWhisperEncoderLayer(nn.Module):
    def __init__(
        self, 
        config: DistilWhisperConfig,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout
        self.ffn_clsr = CLSRFFN(config, 
            uids=config.clsr_langs,
            activation_function=config.activation_function,
            activation_dropout=config.activation_dropout,
            embed_dim=self.embed_dim,
            ffn_dim=config.encoder_ffn_dim
            )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.current_lang = self.is_warm_up = self.step_proportion = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        # Self Attention Pre Norm
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # Self Attention
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # Self Attention Gate
        # Self Attention Residual
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # Self Attention Post Norm - doesnt have

        residual = hidden_states
        # FFN Pre Norm
        hidden_states = self.final_layer_norm(hidden_states)
        # FFN with CLRS
        hidden_states, ffn_gate = self.ffn_clsr(self.current_lang, self.step_proportion, input=hidden_states,
                                                activation_dropout=self.activation_dropout, dropout=self.dropout, training=self.training,
                                                is_warm_up=self.is_warm_up)
        # FFN Residual
        hidden_states = residual + hidden_states
        # FFN Post Norm - doesnt have

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        # Gate values are added at the end of outputs
        outputs += (ffn_gate,)

        return outputs

    def from_whisper(self, src: WhisperEncoderLayer, clsr_langs: list[str]):
        for key, value in src.__dict__['_modules'].items():
            if key not in ["fc1", "fc2"]:
                self.__dict__['_modules'][key] = copy.deepcopy(src.__dict__['_modules'][key])
                assert check_torch_parameters(self.__dict__['_modules'][key], src.__dict__['_modules'][key])
        self.__dict__['_modules']["ffn_clsr"].proj["shared"].set_from_pretrained(fc1=src.__dict__['_modules']["fc1"],
                                                                                 fc2=src.__dict__['_modules']["fc2"],
                                                                                 activation_fn=src.__dict__['_modules']["activation_fn"])
        # Also the language specific one
        for lang in clsr_langs:
            self.__dict__['_modules']["ffn_clsr"].proj[lang].set_from_pretrained(fc1=src.__dict__['_modules']["fc1"],
                                                                                        fc2=src.__dict__['_modules']["fc2"],
                                                                                        activation_fn=src.__dict__['_modules']["activation_fn"])
        assert torch.all(
            self.__dict__['_modules']["ffn_clsr"].proj["shared"].fc1.weight == src.__dict__['_modules']["fc1"].weight)
        assert torch.all(
            self.__dict__['_modules']["ffn_clsr"].proj["shared"].fc1.bias == src.__dict__['_modules']["fc1"].bias)
        assert torch.all(
            self.__dict__['_modules']["ffn_clsr"].proj["shared"].fc2.weight == src.__dict__['_modules']["fc2"].weight)
        assert torch.all(
            self.__dict__['_modules']["ffn_clsr"].proj["shared"].fc2.bias == src.__dict__['_modules']["fc2"].bias)

    def _restart_gates(self):
        self.ffn_clsr._restart_gates()

    def add_language(self, lang, config):
        self.ffn_clsr.add_language(lang, config)



# Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer with MBart->Whisper
class DistilWhisperDecoderLayer(nn.Module):
    def __init__(self, 
        config: DistilWhisperConfig, # TO DO Correct
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.ffn_clsr = CLSRFFN(config, 
            uids=config.clsr_langs,
            activation_function=config.activation_function,
            activation_dropout=config.activation_dropout,
            embed_dim=self.embed_dim,
            ffn_dim=config.decoder_ffn_dim
            )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.current_lang = self.is_warm_up = self.step_proportion = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        #hidden_states = self.activation_fn(self.fc1(hidden_states))
        #hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        #hidden_states = self.fc2(hidden_states)
        #hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states, ffn_gate = self.ffn_clsr(self.current_lang, self.step_proportion, input=hidden_states,
                                                activation_dropout=self.activation_dropout, dropout=self.dropout, training=self.training,
                                                is_warm_up=self.is_warm_up)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)
        
        # Gate values are added at the end of outputs
        outputs += (ffn_gate,)

        return outputs

    def from_whisper(self, src: WhisperDecoderLayer, clsr_langs: list[str]):
        for key, value in src.__dict__['_modules'].items():
            if key not in ["fc1", "fc2"]:
                self.__dict__['_modules'][key] = copy.deepcopy(src.__dict__['_modules'][key])
                assert check_torch_parameters(self.__dict__['_modules'][key], src.__dict__['_modules'][key])
        self.__dict__['_modules']["ffn_clsr"].proj["shared"].set_from_pretrained(fc1=src.__dict__['_modules']["fc1"],
                                                                                 fc2=src.__dict__['_modules']["fc2"],
                                                                                 activation_fn=src.__dict__['_modules']["activation_fn"])
        # Also the language specific one
        for lang in clsr_langs:
            self.__dict__['_modules']["ffn_clsr"].proj[lang].set_from_pretrained(fc1=src.__dict__['_modules']["fc1"],
                                                                                        fc2=src.__dict__['_modules']["fc2"],
                                                                                        activation_fn=src.__dict__['_modules']["activation_fn"])                                                                      
        assert torch.all(
            self.__dict__['_modules']["ffn_clsr"].proj["shared"].fc1.weight == src.__dict__['_modules']["fc1"].weight)
        assert torch.all(
            self.__dict__['_modules']["ffn_clsr"].proj["shared"].fc1.bias == src.__dict__['_modules']["fc1"].bias)
        assert torch.all(
            self.__dict__['_modules']["ffn_clsr"].proj["shared"].fc2.weight == src.__dict__['_modules']["fc2"].weight)
        assert torch.all(
            self.__dict__['_modules']["ffn_clsr"].proj["shared"].fc2.bias == src.__dict__['_modules']["fc2"].bias)

    def _restart_gates(self):
        self.ffn_clsr._restart_gates()

    def add_language(self, lang, config):
        self.ffn_clsr.add_language(lang, config)

class DistilWhisperPreTrainedModel(PreTrainedModel):
    config_class = DistilWhisperConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DistilWhisperEncoderLayer", "DistilWhisperDecoderLayer"] #Changed

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (DistilWhisperDecoder, DistilWhisperEncoder)): #Changed
            module.gradient_checkpointing = value

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

class DistilWhisperEncoder(DistilWhisperPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: DistilWhisperConfig): # To do correct
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList([DistilWhisperEncoderLayer(config) for _ in range(config.encoder_layers)]) # Changed
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_gate_states = ()

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                all_gate_states += (layer_outputs[-1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return CLSRModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions, gate_states=all_gate_states
        )

    def from_whisper(self, src: WhisperEncoder, clsr_langs: list[str]):
        for key, value in src.__dict__.items():
            if not key in ['_modules', "config"]:
                self.__dict__[key] = value
        for key,value in src.__dict__['_modules'].items():
            if not (key == "layers"):
                self.__dict__['_modules'][key] = copy.deepcopy(src.__dict__['_modules'][key])
                assert torch.all(self.__dict__['_modules'][key].weight == src.__dict__['_modules'][key].weight)
                if "bias" in self.__dict__['_modules'][key].state_dict().keys():
                    assert torch.all(self.__dict__['_modules'][key].bias == src.__dict__['_modules'][key].bias)
        for i in range(len(self.layers)):
            self.__dict__['_modules']['layers'][i].from_whisper(src=src.__dict__['_modules']['layers'][i], clsr_langs=clsr_langs)
    
    def _restart_gates(self):
        for i in range(len(self.layers)):
            self.layers[i]._restart_gates()

    def add_language(self, lang, config):
        for i in range(len(self.layers)):
            self.layers[i].add_language(lang, config)

class DistilWhisperDecoder(DistilWhisperPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: DistilWhisperConfig): # To do correct
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList([DistilWhisperDecoderLayer(config) for _ in range(config.decoder_layers)])

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)
        else:
            positions = self.embed_positions(inputs_embeds, past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None
        all_gate_states = ()

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,  # past_key_value
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            all_gate_states += (layer_outputs[-1],)

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return CLSRModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            gate_states=all_gate_states,
        )

    def from_whisper(self, src: WhisperDecoder, clsr_langs: list[str]):
        for key, value in src.__dict__.items():
            if not key in ['_modules', "config"]:
                self.__dict__[key] = value
        for key, value in src.__dict__['_modules'].items():
            if not (key == "layers"):
                self.__dict__['_modules'][key] = copy.deepcopy(src.__dict__['_modules'][key])
                assert torch.all(self.__dict__['_modules'][key].weight == src.__dict__['_modules'][key].weight)
                if "bias" in self.__dict__['_modules'][key].state_dict().keys():
                    assert torch.all(self.__dict__['_modules'][key].bias == src.__dict__['_modules'][key].bias)
        for i in range(len(self.layers)):
            self.__dict__['_modules']['layers'][i].from_whisper(src=src.__dict__['_modules']['layers'][i], clsr_langs=clsr_langs)
    
    def _restart_gates(self):
        for i in range(len(self.layers)):
            self.layers[i]._restart_gates()
    def add_language(self, lang, config):
        for i in range(len(self.layers)):
            self.layers[i].add_language(lang, config)

class DistilWhisperModel(DistilWhisperPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"proj_out.weight"]

    def __init__(self, config: DistilWhisperConfig): # TO DO correct
        super().__init__(config)

        self.encoder = DistilWhisperEncoder(config) # Changed 
        self.decoder = DistilWhisperDecoder(config) # Changed 
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.encoder._freeze_parameters()

    def _mask_input_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.size()

        if self.config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CLSRSeq2SeqOutput]:
        r"""
        Returns:

        Example:
         ```python

         [1, 2, 512]
         ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a CLSRModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, CLSRModelOutput):
            encoder_outputs = CLSRModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                gate_states=encoder_outputs[-1],
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs[:-1] + encoder_outputs[:-1] + ((decoder_outputs[-1], encoder_outputs[-1]),)

        return CLSRSeq2SeqOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            decoder_gate_states=decoder_outputs.gate_states,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_gate_states=encoder_outputs.gate_states,
        )

    def set_language_task(self, language=None, step_proportion=None, is_warm_up=None, task=None):
        if language:
            language = language.lower()
        for layer in self.encoder.layers:
            layer.step_proportion = step_proportion
            layer.is_warm_up = is_warm_up
            if (not language) or (language in layer.ffn_clsr.proj.keys()): # We can set language to existing one or to None
                layer.current_lang = language
            else:
                logger.warning(f"{language} is not present in the encoder")
                break
        for layer in self.decoder.layers:
            layer.step_proportion = step_proportion
            layer.is_warm_up = is_warm_up
            if task == "translate":
                layer.current_lang = None
            elif (not language) or (language in layer.ffn_clsr.proj.keys()): # We can set language to existing one or to None
                layer.current_lang = language
            else:
                logger.warning(f"{language} is not present in the decoder")
                break

    def from_whisper(self, src: WhisperModel):
        self.__dict__['_modules']['encoder'].from_whisper(src=src.__dict__['_modules']['encoder'], clsr_langs=self.config.clsr_langs)
        self.__dict__['_modules']['decoder'].from_whisper(src=src.__dict__['_modules']['decoder'], clsr_langs=self.config.clsr_langs)
    
    def _restart_gates(self):
        self.encoder._restart_gates()
        self.decoder._restart_gates()

    def add_language(self, lang, config=None):
        if not config:
            config = self.config
        self.encoder.add_language(lang, config)
        self.decoder.add_language(lang, config)
    
class DistilWhisperForConditionalGeneration(DistilWhisperPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"encoder.version",
        r"decoder.version",
        r"proj_out.weight",
    ]
    _keys_to_ignore_on_save = [
        r"proj_out.weight",
    ]

    def __init__(self, config: DistilWhisperConfig): # TO DO Correct
        super().__init__(config)
        self.model = DistilWhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        return new_embeddings

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        language=None, # TO DO: remove this naive parameter
    ) -> Union[Tuple[torch.Tensor], CLSRSeq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python

        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            label_smoothing_factor = 0.0
            if label_smoothing_factor != 0.0:
                # Inspired from https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/trainer_pt_utils.py
                logits = lm_logits.view(-1, self.config.vocab_size)
                labels = labels.reshape(-1)
                log_probs = -nn.functional.log_softmax(logits, dim=-1)
                if labels.dim() == log_probs.dim() - 1:
                    labels = labels.unsqueeze(-1)
                padding_mask = labels.eq(-100)
                # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
                # will ignore them in any case.
                labels = torch.clamp(labels, min=0)
                nll_loss = log_probs.gather(dim=-1, index=labels)
                # works for fp16 input tensor too, by internally upcasting it to fp32
                smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

                nll_loss.masked_fill_(padding_mask, 0.0)
                smoothed_loss.masked_fill_(padding_mask, 0.0)

                # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
                num_active_elements = padding_mask.numel() - padding_mask.long().sum()
                nll_loss = nll_loss.sum() / num_active_elements
                smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
                loss =  (1 - label_smoothing_factor) * nll_loss + label_smoothing_factor * smoothed_loss
            else:
                loss_fct = CrossEntropyLoss(label_smoothing=0.1)
                loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
        
        # Gate loss
        gate_loss = None
        # Unwrap gate_states
        if return_dict:
            gate_states = outputs.decoder_gate_states + outputs.encoder_gate_states
        else:
            gate_states = outputs[-1][0]
            gate_states = gate_states[0] + gate_states[1] # Get decoder and encoder
        
        gate_count = gate_sum = 0
        for gate_value in gate_states:
            gate_count += gate_value.numel()
            gate_sum += gate_value.sum().float()
        if self.config.use_gate_budget:
            gate_loss = torch.abs(gate_sum / max(gate_count, 1) - self.config.gate_budget).to(lm_logits)
        else:
            gate_loss = (gate_sum / max(gate_count, 1)).to(lm_logits)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CLSRSeq2SeqLMOutput(
            loss=loss,
            gate_loss=gate_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            decoder_gate_states=outputs.decoder_gate_states,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            encoder_gate_states=outputs.encoder_gate_states,
        )

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        return_timestamps=None,
        task=None,
        language=None,
        lang=None, # TO DO: remove this naive parameter
        is_multilingual=None,
        **kwargs,
    ):
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            return_timestamps (`bool`, *optional*):
                Whether to return the timestamps with the text. This enables the `WhisperTimestampsLogitsProcessor`.
            task (`bool`, *optional*):
                Task to use for generation, either "translate" or "transcribe". The `model.config.forced_decoder_ids`
                will be updated accordingly.
            language (`bool`, *optional*):
                Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. You can
                find all the possible language tokens in the `model.generation_config.lang_to_id` dictionary.
            is_multilingual (`bool`, *optional*):
                Whether or not the model is multilingual.
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchDecoderOnlyOutput`],
                    - [`~generation.SampleDecoderOnlyOutput`],
                    - [`~generation.BeamSearchDecoderOnlyOutput`],
                    - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]
        """
        if generation_config is None:
            generation_config = self.generation_config

        if return_timestamps is not None:
            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You are trying to return timestamps, but the generation config is not properly set."
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`."
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                )

            generation_config.return_timestamps = return_timestamps
        else:
            generation_config.return_timestamps = False

        if language is not None:
            generation_config.language = language
        if task is not None:
            generation_config.task = task

        forced_decoder_ids = []
        if task is not None or language is not None:
            if hasattr(generation_config, "language"):
                if generation_config.language in generation_config.lang_to_id.keys():
                    language_token = generation_config.language
                elif generation_config.language in TO_LANGUAGE_CODE.keys():
                    language_token = f"<|{TO_LANGUAGE_CODE[generation_config.language]}|>"
                else:
                    raise ValueError(
                        f"Unsupported language: {self.language}. Language should be one of:"
                        f" {list(TO_LANGUAGE_CODE.keys()) if generation_config.language in TO_LANGUAGE_CODE.keys() else list(TO_LANGUAGE_CODE.values())}."
                    )
                forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
            else:
                forced_decoder_ids.append((1, None))  # automatically detect the language

            if hasattr(generation_config, "task"):
                if generation_config.task in TASK_IDS:
                    forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                else:
                    raise ValueError(
                        f"The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
                    )
            else:
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))  # defaults to transcribe
            if hasattr(generation_config, "no_timestamps_token_id") and not generation_config.return_timestamps:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        # Legacy code for backward compatibility
        elif hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids is not None:
            forced_decoder_ids = self.config.forced_decoder_ids
        elif (
            hasattr(self.generation_config, "forced_decoder_ids")
            and self.generation_config.forced_decoder_ids is not None
        ):
            forced_decoder_ids = self.generation_config.forced_decoder_ids

        if generation_config.return_timestamps:
            logits_processor = [WhisperTimeStampLogitsProcessor(generation_config)]

        if len(forced_decoder_ids) > 0:
            generation_config.forced_decoder_ids = forced_decoder_ids

        return super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": None,
        }
    def set_language_task(self, language=None, forced_decoder_ids=None, step_proportion=None, is_warm_up=None, task=None):
        if forced_decoder_ids:
            self.config.forced_decoder_ids = forced_decoder_ids
        self.model.set_language_task(language, step_proportion, is_warm_up, task)
    
    def restart_gates(self):
        self.model._restart_gates()

    def add_language(self, lang, config=None):
        if not config:
            config = self.config
        self.model.add_language(lang, config)

    def load_experts(self, target_lang, expert_local_dir=None, **kwargs):
        # TO DO: Check if lang is already loaded
        self.add_language(target_lang)

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        # TO DO: allow for use of safetensors
        # use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

        if expert_local_dir:
            model_path_or_id = expert_local_dir
        else:
            model_path_or_id = self.config._name_or_path # Use base model path (can be local or huggingface hub)
        adapter_state_dict = None

        # 1. Let's first try loading a safetensors adapter weight
        """
        if use_safetensors is not False:
            filepath = DISTILWHISPER_ADAPTER_SAFE_FILE.format(target_lang)

            try:
                weight_path = cached_file(
                    model_path_or_id,
                    filename=filepath,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                )

                adapter_state_dict = safe_load_file(weight_path)

            except EnvironmentError:
                if use_safetensors:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise

            except Exception:
                # For any other exception, we throw a generic error.
                if use_safetensors:
                    raise EnvironmentError(
                        f"Can't load the model for '{model_path_or_id}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a"
                        f" directory containing a file named {filepath}."
                    )
        """
        # 2. If this didn't work let's try loading a PyTorch adapter weight
        if adapter_state_dict is None:
            filepath = DISTILWHISPER_ADAPTER_PT_FILE.format(target_lang)

            try:
                weight_path = cached_file(
                    model_path_or_id,
                    filename=filepath,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    # token=token, # TO DO - uncomment for newer versions of Transformers
                    revision=revision,
                    cache_dir=cache_dir,
                )

                adapter_state_dict = torch.load(
                    weight_path,
                    map_location="cpu",
                    # weights_only=is_torch_greater_or_equal_than_1_13, # TO DO - uncomment this for newer versions of Transformers
                )

            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise

            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the model for '{model_path_or_id}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a"
                    f" directory containing a file named {filepath}."
                )

        model_state_dict = self.state_dict()
        adapter_state_dict = {k: v.to(model_state_dict[k]) for k, v in adapter_state_dict.items()}
        for key, value in adapter_state_dict.items():
            model_state_dict[key] = value
        self.load_state_dict(model_state_dict)
        # TO DO: check what is the difference from doing that above and:
        # self.load_state_dict(adapter_state_dict, strict=False)

        self.set_language_task(language=target_lang)
    #
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

def convert_to_distil_whisper(
        model: WhisperForConditionalGeneration,
        clsr_langs: list[str],
        skip_gate_prob=0.0,
        clsr_gate_dim=64,
        clsr_max_steps=10000,
        clsr_residual=False,
        use_gate_budget=True,
        gate_budget=1.0,
        device=None,
        processor=None,
        sanity_check_language: Optional[str] = None,
        sanity_check_batch: Optional[datasets.Dataset] = None,
):
    config = DistilWhisperConfig.from_whisperconfig(model.config, clsr_langs=clsr_langs, skip_gate_prob=skip_gate_prob, clsr_gate_dim=clsr_gate_dim, clsr_max_steps=clsr_max_steps, clsr_residual=clsr_residual, use_gate_budget=use_gate_budget, gate_budget=gate_budget)
    distilwhisper = DistilWhisperForConditionalGeneration(config)
    for key, value in model.__dict__.items():
        if not key in ['_modules', "config"]:
            distilwhisper.__dict__[key] = value
    distilwhisper.__dict__['_modules']['proj_out'] = copy.deepcopy(model.__dict__['_modules']['proj_out'])
    assert torch.all(distilwhisper.__dict__['_modules']['proj_out'].weight == model.__dict__['_modules']['proj_out'].weight)

    distilwhisper.__dict__['_modules']['model'].from_whisper(src=model.__dict__['_modules']['model'])
    
    # This is a workaround to work on CUDA servers
    # TO DO: fix this problem
    import time
    import os
    model_file_name = "./tmp" + str(time.time())
    distilwhisper.save_pretrained(model_file_name)

    distilwhisper = DistilWhisperForConditionalGeneration.from_pretrained(model_file_name)
    os.system(f"rm -r {model_file_name}")

    if device:
        distilwhisper = distilwhisper.to(device)
    if sanity_check_batch:
        sanity_check_language = sanity_check_language or clsr_langs[0]
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=sanity_check_language,
                                                                                task="transcribe")
        distilwhisper.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=sanity_check_language,
                                                                                task="transcribe")
        distilwhisper.set_language_task(language=None, step_proportion=0.0, is_warm_up=False)
        with torch.no_grad():
            generated_tokens = model.generate(
                    input_features=sanity_check_batch["input_features"].to(device=model.device),
                    max_new_tokens=255,
                ).cpu().numpy()
            generated_tokens_distill = distilwhisper.generate(
                    input_features=sanity_check_batch["input_features"].to(device=distilwhisper.device),
                    max_new_tokens=255,
                ).cpu().numpy()
            
            pred_str1 = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            pred_str2 = processor.tokenizer.batch_decode(generated_tokens_distill, skip_special_tokens=True)

        assert np.array_equal(generated_tokens, generated_tokens_distill)


    return distilwhisper


