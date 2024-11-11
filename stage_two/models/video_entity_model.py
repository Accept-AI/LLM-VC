import torch
import math
from torch import nn
import copy
# config:  {
#   "attention_probs_dropout_prob": 0.1,
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "hidden_size": 768,
#   "down_dim": 256,
#   "initializer_range": 0.02,
#   "intermediate_size": 3072,
#   "max_target_embeddings": 512,
#   "num_attention_heads": 12,
#   "num_decoder_layers": 3,
#   "num_hidden_layers": 12,
#   "type_vocab_size": 2,
#   "vocab_size": 30522
# }
class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        # if not isinstance(config, PretrainedConfig):
        #     raise ValueError(
        #         "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
        #         "To create a model from a Google pretrained model use "
        #         "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
        #             self.__class__.__name__, self.__class__.__name__
        #         ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        # if prefix is None and (task_config is None or task_config.local_rank == 0):
        #     logger.info("-" * 20)
        #     if len(missing_keys) > 0:
        #         logger.info("Weights of {} not initialized from pretrained model: {}"
        #                     .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
        #     if len(unexpected_keys) > 0:
        #         logger.info("Weights from pretrained model not used in {}: {}"
        #                     .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
        #     if len(error_msgs) > 0:
        #         logger.error("Weights from pretrained model cause errors in {}: {}"
        #                      .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
class BertSelfOutput_low(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput_low, self).__init__()
        self.dense = nn.Linear(config.down_dim, config.down_dim)
        self.LayerNorm = LayerNorm(config.down_dim, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        #print("hidden:states: ", hidden_states.shape)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        #print("self.intermediate_act_fn: ", self.intermediate_act_fn)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attention_mask):
        #print("attention_mask: ", attention_mask.shape)
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #print("attention_scores: ", attention_scores.shape)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        #print("attention_mask: ", attention_mask.shape)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_scores
class MultiHeadAttention_low(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, config):
        super(MultiHeadAttention_low, self).__init__()

        # if config.hidden_size % config.num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads_low = config.num_attention_heads_low
        self.attention_head_size = int(config.down_dim / config.num_attention_heads_low)
        self.all_head_size = self.num_attention_heads_low * self.attention_head_size

        self.query = nn.Linear(config.down_dim, self.all_head_size)
        self.key = nn.Linear(config.down_dim, self.all_head_size)
        self.value = nn.Linear(config.down_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads_low, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, attention_mask):
        #print("attention_mask: ", attention_mask.shape)
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #print("attention_scores: ", attention_scores.shape)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_scores

class DecoderAttention(nn.Module):
    def __init__(self, config):
        super(DecoderAttention, self).__init__()
        self.att = MultiHeadAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, q, k, v, attention_mask):
        att_output, attention_probs = self.att(q, k, v, attention_mask)
        attention_output = self.output(att_output, q)
        return attention_output, attention_probs

class DecoderAttention_low(nn.Module):
    def __init__(self, config):
        super(DecoderAttention_low, self).__init__()
        self.att = MultiHeadAttention_low(config)
        self.output = BertSelfOutput_low(config)

    def forward(self, q, k, v, attention_mask):
        att_output, attention_probs = self.att(q, k, v, attention_mask)

        #print("att_output: ", att_output.shape)
        #print("q: ", q.shape)
        attention_output = self.output(att_output, q)
        return attention_output, attention_probs

class VELayer(nn.Module):
    def __init__(self, config):
        super(VELayer, self).__init__()
        self.LayerNorm_v = LayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm_e = LayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm_v_down = LayerNorm(config.down_dim, eps=1e-12)
        self.LayerNorm_e_down = LayerNorm(config.down_dim, eps=1e-12)
        self.slf_video_attn = DecoderAttention(config)
        self.slf_entity_attn = DecoderAttention(config)
        self.video_down_proj = nn.Linear(config.hidden_size, config.down_dim)
        self.entity_down_proj = nn.Linear(config.hidden_size, config.down_dim)
        self.video_up_proj = nn.Linear(config.down_dim, config.hidden_size)
        self.entity_up_proj = nn.Linear(config.down_dim, config.hidden_size)
        # low dim
        self.cross_videoQ_entity = DecoderAttention_low(config)
        self.cross_entityQ_video = DecoderAttention_low(config)

        self.video_intermediate = BertIntermediate(config)
        self.video_output = BertOutput(config)

        self.entity_intermediate = BertIntermediate(config)
        self.entity_output = BertOutput(config)

    def forward(self, video_input, entity_input, video_mask=None, entity_mask=None):
        #print("dec_input: ", dec_input.shape)  # dec_input:  torch.Size([1, 30, 768])
        video_input = self.LayerNorm_v(video_input)
        entity_input = self.LayerNorm_e(entity_input)
        video_self_out, _ = self.slf_video_attn(video_input, video_input, video_input, video_mask)
        entity_self_out, _ = self.slf_entity_attn(entity_input, entity_input, entity_input, entity_mask)
        video_down = self.video_down_proj(video_self_out)
        entity_down = self.entity_down_proj(entity_self_out)
        video_down = self.LayerNorm_v_down(video_down)
        entity_down = self.LayerNorm_e_down(entity_down)
        # low dim
        video_cross_out, _ = self.cross_videoQ_entity(video_down, entity_down, entity_down, entity_mask)
        entity_cross_out, _ = self.cross_entityQ_video(entity_down, video_down, video_down, video_mask)
        # up dim
        video_up = self.video_up_proj(video_cross_out)
        entity_up = self.entity_up_proj(entity_cross_out)

        video_intermediate_output = self.video_intermediate(video_up)
        video_output = self.video_output(video_intermediate_output, video_up)

        entity_intermediate_output = self.entity_intermediate(entity_up)
        entity_output = self.entity_output(entity_intermediate_output, entity_up)
        return video_output, entity_output

class VEInter(nn.Module):
    def __init__(self, config):
        super(VEInter, self).__init__()
        layer = VELayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_ve_layers)])

    def forward(self, video_input, entity_input, video_mask=None, entity_mask=None):

        for layer_module in self.layer:
            video_input, entity_input = layer_module(video_input, entity_input, video_mask, entity_mask)

        video_output = video_input
        entity_output = entity_input
        return video_output, entity_output
