from functools import partial
import torch
import torch.nn as nn
import argparse
from flash_attn.utils.generation import GenerationMixin
from flash_attn.utils.distributed import sync_shared_params

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

# grab all functions / modules from long_conv_lm.py
from src.models.sequence.long_conv_lm import LMBackbone
from src.models.sequence.long_conv_lm import _init_weights
from transformers import BertModel,BertConfig
from transformers import AutoTokenizer, AutoModel
from src.utils.genslm.inference import GenSLM
class DNAEmbeddingGENSLM(nn.Module, GenerationMixin):
    """DNA Embedding Model, which is the same as ConvLMHeadModel (in long_conv_lm.py), except no decoder head, we just pass back the hidden states for downstream tasks."""

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 process_group=None, layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1, dropout_cls=nn.Dropout,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,
                 fused_mlp=False, fused_dropout_add_ln=False, residual_in_fp32=False,
                 pad_vocab_size_multiple: int = 1, sequence_parallel=True,
                 device=None, dtype=None, return_hidden_state=False, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model  # for decoder
        self.process_group = process_group
        self.return_hidden_state = return_hidden_state
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        # self.backbone = LMBackbone(
        #     d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
        #     process_group=process_group,
        #     layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
        #     max_position_embeddings=max_position_embeddings,
        #     resid_dropout=resid_dropout, embed_dropout=embed_dropout,
        #     dropout_cls=dropout_cls, layer_norm_epsilon=layer_norm_epsilon,
        #     initializer_cfg=initializer_cfg, fused_mlp=fused_mlp,
        #     fused_dropout_add_ln=fused_dropout_add_ln, residual_in_fp32=residual_in_fp32,
        #     sequence_parallel=sequence_parallel,
        #     **factory_kwargs, **kwargs
        # )
        parser = argparse.ArgumentParser()
        args=parser.parse_args()
        args.config_name = '/usr/data/3-new-12w-0'
        num_labels = 2
        args.task_name='cola'
        args.cache_dir = '/usr/data/3-new-12w-0'
        args.hidden_dropout_prob = 0.1
        args.attention_probs_dropout_prob = 0.1
        args.max_seq_length = 1024
        args.rnn = 'lstm'
        args.num_rnn_layer = 2
        args.rnn_dropout = 0.1
        args.rnn_hidden = 768
        args.do_lower_case = False
        args.model_type = 'bert'

        config = BertConfig.from_pretrained(
            args.config_name,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        if args.model_type in ["dnalong", "dnalongcat"]:
            assert args.max_seq_length % 512 == 0
        config.split = int(args.max_seq_length/512)
        config.rnn = args.rnn
        config.num_rnn_layer = args.num_rnn_layer
        config.rnn_dropout = args.rnn_dropout
        config.rnn_hidden = args.rnn_hidden
        # self.backbone=BertModel(config).from_pretrained('/usr/data/3-new-12w-0')
        self.backbone=GenSLM("genslm_25M_patric", model_cache_dir="/usr/data/genslm/weight")
        if process_group is None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        else:
            if ColumnParallelLinear is None:
                raise ImportError('fused_dense_lib is not installed')
            self.lm_head = ColumnParallelLinear(
                d_model, vocab_size, process_group, bias=False,
                sequence_parallel=sequence_parallel, **factory_kwargs
            )
        # Initialize weights and apply final processing
        # self.apply(partial(_init_weights, n_layer=n_layer,
        #                    **(initializer_cfg if initializer_cfg is not None else {})))
        # self.tie_weights()

    def tie_weights(self):
        # self.lm_head.weight = self.backbone.model.embeddings.word_embeddings.weight
        # if self.process_group is not None:
        #     sync_shared_params(self, self.process_group)

        print("no need to tie weights for GenSLM")

    def forward(self, input_ids, mask=None,position_ids=None, inference_params=None, state=None): # state for the repo interface
        #return pooled output
        hidden_states = self.backbone(input_ids=input_ids, attention_mask=mask,output_hidden_states=True,).hidden_states[-1]
        # we only need the last hidden state for embeddings (decoder head will predict classification task)
        return hidden_states, None

    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model


def load_backbone(model, state_dict, freeze_backbone=False, ignore_head=True):
    """

    Modifies state dict loading with custom function.  This is necessary because the head of
    a lm outputs logits for vocab, but we just the embeddings for downstream tasks.

    inputs:
        model: nn.Module, the from 'scratch' model
        state_dict: dict, from the pretrained weights
        ignore_head: bool, whether to inflate weights in the head (or keep scratch weights).
            If number of classes changes (eg, imagenet to hmdb51), then you need to use this.

    return:
        state_dict: dict, update with inflated weights
    """

    # consumes prefix from pretrained model, if necessary
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, "model."
    )

    model_new_params_dict = model.state_dict()
    updated_model_state_dict = {}

    # loop through scratch model keys (pretrained may have extra stuff)
    for key in sorted(model_new_params_dict.keys()):

        loaded_params = state_dict.get(key, None)
        # make sure key is in the loaded params first, if not, then print it out
    
        if loaded_params is None:
            # This should never happen, it should be there!
            print("Missing key in pretrained model!", key)
            raise Exception

        elif ignore_head and 'head' in key:
            # ignore head weights
            print("found head key / parameter, load from scratch", key)
            # using scratch by default, nothing needed
            used_params = model_new_params_dict[key]

        elif "decoder" in key:
            print("found decoder key / parameter, load from scratch", key)
            used_params = model_new_params_dict[key]
        else:
            print('key: shape MATCH, loading', key)  # load matched weights
            used_params = loaded_params

        # we need to pass back a state dict with the '.model' prefix!!!!!
        key_with_prefix = 'model.' + key
        updated_model_state_dict[key_with_prefix] = used_params

    if freeze_backbone:
        print("freezing model backbone params!")
        # note, decoder not included in backbone
        for name, param in model.named_parameters():
            param.requires_grad = False

    # we have updated the new model state dict with pretrained now
    return updated_model_state_dict