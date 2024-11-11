from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, Qwen2ForCausalLM
# mine
#from transformers import LlamaForCausalLM_mine
import torch
from torch import nn
import einops
import contextlib
from .Qformer import BertConfig, BertLMHeadModel
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from typing import List
import pickle as pkl
import sys
import io
from .video_entity_model import VEInter
import argparse
import copy
# VE_config = {
#             "attention_probs_dropout_prob": 0.1,
#             "hidden_act": "gelu",
#             "hidden_dropout_prob": 0.1,
#             "hidden_size": 768,
#             "down_dim": 256,
#             "initializer_range": 0.02,
#             "intermediate_size": 768,
#             "num_attention_heads": 12,
#             "num_attention_heads_low": 12,
#             "num_ve_layers": 2,
#             "num_hidden_layers": 12,
#         }
parser = argparse.ArgumentParser(description='VE module setting')
parser.add_argument('--attention_probs_dropout_prob', default=0.1, type=float, help='attention_probs_dropout_prob')
parser.add_argument('--hidden_act', default="gelu", type=str, help='hidden_act')
parser.add_argument('--hidden_dropout_prob', default=0.1, type=float, help='hidden_dropout_prob')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden_size')
parser.add_argument('--down_dim', default=512, type=int, help='down_dim')
parser.add_argument('--initializer_range', default=0.02, type=float, help='initializer_range')
parser.add_argument('--intermediate_size', default=768, type=int, help='intermediate_size')
parser.add_argument('--num_attention_heads', default=12, type=int, help='num_attention_heads')
parser.add_argument('--num_attention_heads_low', default=8, type=int, help='num_attention_heads_low')
parser.add_argument('--num_ve_layers', default=1, type=int, help='num_ve_layers')
VE_config = parser.parse_args()

def process_output_tokens(predict_model, tokens):
    output_texts = []
    for output_token in tokens:
        output_text = predict_model.tokenizer.decode(output_token)
        end_token_index = output_text.find('<|end_of_text|>')
        if end_token_index != -1:
            output_text = output_text[:end_token_index]
        output_texts.append(output_text)
    return output_texts


class RestrictTokenGenerationLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_id_list: List[int]):
        super().__init__()
        self.allowed_token_id_list = allowed_token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -float('inf'))
        for allowed_id in self.allowed_token_id_list:
            mask[:, allowed_id] = scores[:, allowed_id]
        return mask


class LLM_Captioner(nn.Module):
    def __init__(self,
                 # LLM part
                 llm_ckpt="/home/xzy/xzy_nba/meta-llama/Llama-3.2-1B",
                 tokenizer_ckpt="/home/xzy/xzy_nba/meta-llama/Llama-3.2-1B",
                 # Q-former part
                 max_frame_pos=128,
                 num_video_query_token=32,
                 num_features=768,
                 device="cuda:1",
                 inference=False,
                 **kwargs,
                 ):
        super().__init__()
        if len(kwargs):
            print(f'kwargs not used: {kwargs}')

        self.device = device
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
        self.llama_model = LlamaForCausalLM.from_pretrained(llm_ckpt, torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.tokenizer))
        self.ln_vision = LayerNorm(num_features)
        self.num_video_query_token = num_video_query_token
        self.inference = inference
        self.tokenizer.pad_token_id = 128001
        # Initialize video Q-former
        self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(num_query_token=num_video_query_token,
                                                                              vision_width=num_features,
                                                                              num_hidden_layers=1)
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # llama projection
        self.llama_proj = nn.Linear(
            self.video_Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.video_out_proj = nn.Linear(
            768, self.llama_model.config.hidden_size
        ).to(self.device)
        self.entity_out_proj = nn.Linear(
            768, self.llama_model.config.hidden_size
        ).to(self.device)

        # prompt projection
        self.entity_one_proj = nn.Linear(num_features, self.llama_model.config.hidden_size).to(self.device)
        self.entity_two_proj = nn.Linear(num_features, self.llama_model.config.hidden_size).to(self.device)
        # video frame positional embedding
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, num_features)


        # move to device
        self.VEInter = VEInter(VE_config).to(self.device)   # video-entity interaction

        self.llama_model = self.llama_model.to(self.device)
        for name, param in self.llama_model.named_parameters():
            # if name.split('.')[2] in ["0", "1", "2", "3", "4", "5", "6", "7"]:#, "4", "5", "6", "7"]:
            #     param.requires_grad = True
            # else:
            param.requires_grad = False
            #
            #     print("name-4", name)
            # print(name.split('.')[2])
            # if any(f'model.layers.{i}' in name for i in range(2)):
            #     #print("---name: ", name)
            #     param.requires_grad = True
            # else:
            #     param.requires_grad = False
        self.video_Qformer = self.video_Qformer.to(self.device)
        self.llama_proj = self.llama_proj.to(self.device)
        self.ln_vision = self.ln_vision.to(self.device)
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()
        self.video_frame_position_embedding = self.video_frame_position_embedding.to(self.device)

        # Here is a trick for inference that generates soccer relevant, you can delete this LogitsProcessorList part (including in generation function)
        file_path = '/home/xzy/xzy_nba/LLM_VC/Player_identify/code/B_data_preprocessing/C_VG_nba_llama_3-2_3B.pkl'
        with open(file_path, 'rb') as file:
            self.token_ids_list = pkl.load(file)
        # self.token_ids_list.append(128000) # llama3
        # self.token_ids_list.append(128001)
        # self.token_ids_list.append(151644)
        # self.token_ids_list.append(151645)
        self.processor = RestrictTokenGenerationLogitsProcessor(allowed_token_id_list=self.token_ids_list)
        self.logits_prosessors = LogitsProcessorList()
        self.logits_prosessors.append(self.processor)

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        #print("num_query_token")
        encoder_config = BertConfig.from_pretrained("/home/xzy/xzy_nba/bert-base-uncased")
        #print("encoder_config: ", encoder_config)
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def maybe_autocast(self, dtype=torch.float16):
        # enable_autocast = self.cuda() != torch.device("cpu")
        # if enable_autocast:
        #     return torch.cuda.amp.autocast(dtype=dtype)
        return torch.cuda.amp.autocast(dtype=dtype)
        # else:
        #     return contextlib.nullcontext()

    #def forward(self, samples, validating=False):
        # with VE module
    def forward(self, video, video_mask, caption_tokens, labels, attention_mask, name_one_embeds,
                entity_one_feature, name_two_embeds, entity_two_feature, validating=False):

        video_features = video.to(self.device)
        # print("video: ", video_features.shape)  # torch.Size([2, 1, 60, 768])
        targets = labels.to(self.device)
        atts_llama = attention_mask.to(self.device)

        # print("atts_llama: ", atts_llama.shape)  # torch.Size([2, 30])
        inputs_ids = caption_tokens.to(self.device)
        # print(samples["caption_info"])
        batch_size = None
        time_length = None

        batch_size, _, time_length, _ = video_features.size()

        # entity_mask
        entity_mask = torch.ones(batch_size, 2, dtype=atts_llama.dtype).to(self.device)  # torch.Size([2, 2])
        entity_mask = entity_mask.view(batch_size,1,1,2)
        video_mask = video_mask.view(batch_size, -1).to(self.device)  #  torch.Size([2, 60])
        video_mask = video_mask.view(batch_size,1,1,60)


        if len(video_features.size()) != 4:
            video_features = video_features.unsqueeze(-2)
        # print("pre: ", video_features.shape)
        video_input = copy.deepcopy(video_features)
        video_features = self.ln_vision(video_features)  # torch.Size([2, 1, 60, 768])  ln_vision : Layernorm

        #video_features = video_features.view(batch_size, time_length, -1)  # 2,60,768
        # print("video_features: ", video_features.shape)
        # print("name_one_embeds: ", name_one_embeds.shape)  # torch.Size([2, 3072])
        # print("entity_one_feature: ", entity_one_feature.shape)  # torch.Size([2, 1, 768])
        # print("name_two_embeds: ", name_two_embeds.shape)  # torch.Size([2, 3072])
        # print("entity_two_feature: ", entity_two_feature.shape)  # torch.Size([2, 1, 768])

        # print("after: ", video_features.shape)
        ##########
        video_features = einops.rearrange(video_features, 'b n t f -> (b t) n f', b=batch_size, t=time_length)  # 2,60,768
        # print("time_length: ", time_length)  # 60
        # print("video_features_rearrange: ", video_features.shape)  # torch.Size([120, 1, 768])


        video_input = video_input.view(batch_size, time_length, -1)
        #print("video_input: ", video_input.shape)
        entity_input = torch.cat((entity_one_feature.to(self.device), entity_two_feature.to(self.device)), dim=1)

        position_ids = torch.arange(time_length, dtype=torch.long).to(self.device)
        # print("position_ids: ", position_ids.shape)  #  torch.Size([60])
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        # print("position_ids_unsqueeze: ", position_ids.shape)
        frame_position_embeddings = self.video_frame_position_embedding(position_ids)
        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        # print("frame_position_embeddings: ", frame_position_embeddings.shape)
        frame_hidden_state = einops.rearrange(video_features, '(b t) n f -> b t n f', b=batch_size, t=time_length)
        # print("frame_hidden_state: ", frame_hidden_state.shape)
        frame_hidden_state = frame_position_embeddings + frame_hidden_state

        frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size,
                                              t=time_length)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(
            frame_hidden_state.to(self.device))
        video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1).to(
            frame_hidden_state.to(self.device))
        # print("video_query_tokens: ", video_query_tokens.shape)
        video_query_output = self.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
        )
        video_hidden = video_query_output.last_hidden_state
        # print("video_hidden: ", video_hidden.shape)  # torch.Size([2, 32, 768])
        #####
        inputs_llama = self.llama_proj(video_hidden)  # torch.Size([2, 32, 3072])

        # VE module
        video_out, entity_out = self.VEInter(video_input, entity_input, video_mask, entity_mask)
        input_video_llama = self.video_out_proj(video_out)
        input_entity_llama = self.entity_out_proj(entity_out)
        #print("video_out: ", video_out.shape)
#         print("entity_out: ", entity_out.shape)   video_out:  torch.Size([2, 60, 768])
#                                                   entity_out:  torch.Size([2, 2, 768])



        if validating:
            return self.generate_text(inputs_llama, name_one_embeds, entity_one_feature, name_two_embeds,
                                      entity_two_feature, input_entity_llama, input_video_llama)
            # return temp_res_text, anonymized

        # without name prompt
        visual_label = torch.full((batch_size, self.num_video_query_token), -100, dtype=targets.dtype)
        # revise: self.num_video_query_token+14 (full name)
        #visual_label = torch.full((batch_size, self.num_video_query_token + 2 + 60 + 18), -100, dtype=targets.dtype)
        # visual_label = torch.full((batch_size, 60 + 18), -100, dtype=targets.dtype)  # without qformer

        # print("visual_label: ", visual_label.shape)  # torch.Size([2, 32])
        # print(visual_label)  # full of -100
        concat_targets = torch.cat((visual_label.to(self.device), targets.to(self.device)), dim=1)
        temp_input_ids = inputs_ids.clone().to(self.device)
        targets_embeds = self.llama_model.model.embed_tokens(temp_input_ids)
        #print("targets_embeds: ", targets_embeds.shape)  # torch.Size([32, 30, 2048])
        # add
        # self.entity_one_proj = nn.Linear(num_features, self.llama_model.config.hidden_size)
        # self.entity_two_proj = nn.Linear(num_features, self.llama_model.config.hidden_size)
        entity_one_feature = self.entity_one_proj(entity_one_feature.to(self.device))
        entity_two_feature = self.entity_two_proj(entity_two_feature.to(self.device))
        # print("entity_two_feature: ", entity_two_feature.shape)  # entity_two_feature:  torch.Size([2, 1, 3072])

        name_one_embeds = name_one_embeds.view(batch_size, 8, -1).to(
            self.device)  # name_one_embeds:  torch.Size([2, 1, 3072])
        name_two_embeds = name_two_embeds.view(batch_size, 8, -1).to(self.device)
        # print("name_two_embeds: ", name_two_embeds.shape)

        # ori
        embedding_cat = torch.cat((inputs_llama, targets_embeds), dim=1)
        # revise  inputs_llama, video_out, entity_out, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature,
        # embedding_cat = torch.cat((inputs_llama, input_video_llama, input_entity_llama, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature,
        #                            targets_embeds), dim=1)
        # print("embedding_cat: ", embedding_cat.shape)
        # without prompt
        mask_prefix = torch.ones(batch_size, self.num_video_query_token, dtype=atts_llama.dtype)
        #mask_prefix = torch.ones(batch_size, self.num_video_query_token + 2 + 60 + 18, dtype=atts_llama.dtype)  #
        # mask_prefix = torch.ones(batch_size, 60 + 18, dtype=atts_llama.dtype)  # without qformer

        # print("mask_prefix: ", mask_prefix.shape)  # torch.Size([2, 36])
        mask = torch.concat((mask_prefix.to(self.device), atts_llama.to(self.device)), dim=1)

        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=embedding_cat,
                attention_mask=mask,
                return_dict=True,
                labels=concat_targets,
            )
        sys.stdout = original_stdout
        loss = outputs.loss
        # print("loss: ", loss)
        return loss

    def generate_text(self, inputs_llama, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature, entity_out, video_out):
        batch_size = inputs_llama.size(0)
        # add
        name_one_embeds = name_one_embeds.view(batch_size, 8, -1).to(
            self.device)  # name_one_embeds:  torch.Size([2, 1, 3072])
        name_two_embeds = name_two_embeds.view(batch_size, 8, -1).to(self.device)
        entity_one_feature = self.entity_one_proj(entity_one_feature.to(self.device))
        entity_two_feature = self.entity_two_proj(entity_two_feature.to(self.device))

        start_embeds = self.llama_model.model.embed_tokens(torch.tensor([128000]).to(self.device))  # llama3
        # start_embeds = self.llama_model.model.embed_tokens(torch.tensor([151645]).cuda())
        # inputs_llama_with_s = torch.cat(
        #     [inputs_llama, video_out, entity_out, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature,
        #      start_embeds.expand(inputs_llama.size(0), -1, -1)], dim=1).to(
        #     dtype=torch.bfloat16)
        inputs_llama_with_s = torch.cat(
            [inputs_llama,
             start_embeds.expand(inputs_llama.size(0), -1, -1)], dim=1).to(
            dtype=torch.bfloat16)
        #print("inputs_llama_with_s: ", inputs_llama_with_s.shape)  # inputs_llama_with_s:  torch.Size([2, 33, 3072])
        # without prompt
        mask_prefix = torch.ones(batch_size, self.num_video_query_token + 1, dtype=torch.bool).to(self.device)
        #mask_prefix = torch.ones(batch_size, self.num_video_query_token+1+18+2+60, dtype=torch.bool).to(self.device)
        # mask_prefix = torch.ones(batch_size, 60 + 1 + 18, dtype=torch.bool).to(self.device) # without qformer

        temp_res_tokens = self.llama_model.generate(
            logits_processor=self.logits_prosessors,
            renormalize_logits=True,
            inputs_embeds=inputs_llama_with_s,
            attention_mask=mask_prefix,
            max_new_tokens=128,
            num_beams=5,
            do_sample=True,
            min_length=5,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
            pad_token_id=128001,
        )
        res_text = process_output_tokens(self, temp_res_tokens)
        return res_text
#     # without VE module
#     def forward(self, video, video_mask, caption_tokens, labels, attention_mask, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature, validating=False):
#
#         video_features = video.to(self.device)
#         #print("video: ", video_features.shape)  # torch.Size([2, 1, 60, 768])
#         targets = labels.to(self.device)
#         atts_llama = attention_mask.to(self.device)
#
#         #print("atts_llama: ", atts_llama.shape)  # torch.Size([2, 30])
#         inputs_ids = caption_tokens.to(self.device)
#         # print(samples["caption_info"])
#         batch_size = None
#         time_length = None
#
#         batch_size, _, time_length, _ = video_features.size()
#
#         # entity_mask
#         entity_mask = torch.ones(batch_size, 2, dtype=atts_llama.dtype)
#
#         if len(video_features.size()) != 4:
#             video_features = video_features.unsqueeze(-2)
#         #print("pre: ", video_features.shape)
#         video_features = self.ln_vision(video_features)   # torch.Size([2, 1, 60, 768])  ln_vision : Layernorm
#         video_features = video_features.view(batch_size, time_length, -1)
#         # print("video_features: ", video_features.shape)
#         # print("name_one_embeds: ", name_one_embeds.shape)  # torch.Size([2, 3072])
#         # print("entity_one_feature: ", entity_one_feature.shape)  # torch.Size([2, 1, 768])
#         # print("name_two_embeds: ", name_two_embeds.shape)  # torch.Size([2, 3072])
#         # print("entity_two_feature: ", entity_two_feature.shape)  # torch.Size([2, 1, 768])
#
#         #print("after: ", video_features.shape)
# ##########
#         video_features = einops.rearrange(video_features, 'b n t f -> (b t) n f', b=batch_size, t=time_length)
#         # print("time_length: ", time_length)  # 60
#         # print("video_features_rearrange: ", video_features.shape)  # torch.Size([120, 1, 768])
#
#         position_ids = torch.arange(time_length, dtype=torch.long).to(self.device)
#         #print("position_ids: ", position_ids.shape)  #  torch.Size([60])
#         position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
#         #print("position_ids_unsqueeze: ", position_ids.shape)
#         frame_position_embeddings = self.video_frame_position_embedding(position_ids)
#         frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
#         #print("frame_position_embeddings: ", frame_position_embeddings.shape)
#         frame_hidden_state = einops.rearrange(video_features, '(b t) n f -> b t n f', b=batch_size, t=time_length)
#         #print("frame_hidden_state: ", frame_hidden_state.shape)
#         frame_hidden_state = frame_position_embeddings + frame_hidden_state
#
#         frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size, t=time_length)
#         frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(frame_hidden_state.to(self.device))
#         video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1).to(
#             frame_hidden_state.to(self.device))
#         #print("video_query_tokens: ", video_query_tokens.shape)
#         video_query_output = self.video_Qformer.bert(
#             query_embeds=video_query_tokens,
#             encoder_hidden_states=frame_hidden_state,
#             encoder_attention_mask=frame_atts,
#             return_dict=True,
#         )
#         video_hidden = video_query_output.last_hidden_state
#         #print("video_hidden: ", video_hidden.shape)  # torch.Size([2, 32, 768])
# #####
#         inputs_llama = self.llama_proj(video_hidden)  # torch.Size([2, 32, 3072])
#         #inputs_llama = self.llama_proj(video_features)  # torch.Size([2, 32, 3072])
#
#         #print("inputs_llama: ", inputs_llama.shape)
#
#         if validating:
#             return self.generate_text(inputs_llama, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature)
#             # return temp_res_text, anonymized
#
#         # without name prompt
#         #visual_label = torch.full((batch_size, self.num_video_query_token), -100, dtype=targets.dtype)
#         # revise: self.num_video_query_token+14 (full name)
#         visual_label = torch.full((batch_size, self.num_video_query_token + 18), -100, dtype=targets.dtype)
#         #visual_label = torch.full((batch_size, 60 + 18), -100, dtype=targets.dtype)  # without qformer
#
#         #print("visual_label: ", visual_label.shape)  # torch.Size([2, 32])
#         #print(visual_label)  # full of -100
#         concat_targets = torch.cat((visual_label.to(self.device), targets.to(self.device)), dim=1)
#         temp_input_ids = inputs_ids.clone().to(self.device)
#         targets_embeds = self.llama_model.model.embed_tokens(temp_input_ids)
#         # add
#         # self.entity_one_proj = nn.Linear(num_features, self.llama_model.config.hidden_size)
#         # self.entity_two_proj = nn.Linear(num_features, self.llama_model.config.hidden_size)
#         entity_one_feature = self.entity_one_proj(entity_one_feature.to(self.device))
#         entity_two_feature = self.entity_two_proj(entity_two_feature.to(self.device))
#         #print("entity_two_feature: ", entity_two_feature.shape)  # entity_two_feature:  torch.Size([2, 1, 3072])
#
#         name_one_embeds = name_one_embeds.view(batch_size, 8, -1).to(self.device)  # name_one_embeds:  torch.Size([2, 1, 3072])
#         name_two_embeds = name_two_embeds.view(batch_size, 8, -1).to(self.device)
#         #print("name_two_embeds: ", name_two_embeds.shape)
#
#
#         # ori
#         #embedding_cat = torch.cat((inputs_llama, targets_embeds), dim=1)
#         # revise
#         embedding_cat = torch.cat((name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature, inputs_llama, targets_embeds), dim=1)
#         #print("embedding_cat: ", embedding_cat.shape)
#         mask_prefix = torch.ones(batch_size, self.num_video_query_token + 18, dtype=atts_llama.dtype)
#         # mask_prefix = torch.ones(batch_size, 60 + 18, dtype=atts_llama.dtype)  # without qformer
#         #mask_prefix = torch.ones(batch_size, self.num_video_query_token, dtype=atts_llama.dtype)
#         #print("mask_prefix: ", mask_prefix.shape)  # torch.Size([2, 36])
#         mask = torch.concat((mask_prefix.to(self.device), atts_llama.to(self.device)), dim=1)
#
#         original_stdout = sys.stdout
#         sys.stdout = io.StringIO()
#         with self.maybe_autocast():
#             outputs = self.llama_model(
#                 inputs_embeds=embedding_cat,
#                 attention_mask=mask,
#                 return_dict=True,
#                 labels=concat_targets,
#             )
#         sys.stdout = original_stdout
#         loss = outputs.loss
#         #print("loss: ", loss)
#         return loss
#
#     def generate_text(self, inputs_llama, name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature):
#         batch_size = inputs_llama.size(0)
#         # add
#         name_one_embeds = name_one_embeds.view(batch_size, 8, -1).to(self.device)  # name_one_embeds:  torch.Size([2, 1, 3072])
#         name_two_embeds = name_two_embeds.view(batch_size, 8, -1).to(self.device)
#         entity_one_feature = self.entity_one_proj(entity_one_feature.to(self.device))
#         entity_two_feature = self.entity_two_proj(entity_two_feature.to(self.device))
#
#         start_embeds = self.llama_model.model.embed_tokens(torch.tensor([128000]).to(self.device)) # llama3
#         #start_embeds = self.llama_model.model.embed_tokens(torch.tensor([151645]).cuda())
#         inputs_llama_with_s = torch.cat([name_one_embeds, entity_one_feature, name_two_embeds, entity_two_feature, inputs_llama, start_embeds.expand(inputs_llama.size(0), -1, -1)], dim=1).to(
#             dtype=torch.bfloat16)
#         # inputs_llama_with_s = torch.cat(
#         #     [inputs_llama,
#         #      start_embeds.expand(inputs_llama.size(0), -1, -1)], dim=1).to(
#         #     dtype=torch.bfloat16)
#         # print("inputs_llama_with_s: ", inputs_llama_with_s.shape)  # inputs_llama_with_s:  torch.Size([2, 33, 3072])
#         #mask_prefix = torch.ones(batch_size, self.num_video_query_token+1+18, dtype=torch.bool).to(self.device)
#         mask_prefix = torch.ones(batch_size, 60 + 1 + 18, dtype=torch.bool).to(self.device)
#         #mask_prefix = torch.ones(batch_size, self.num_video_query_token + 1, dtype=torch.bool).to(self.device)
#         temp_res_tokens = self.llama_model.generate(
#             logits_processor=self.logits_prosessors,
#             renormalize_logits=True,
#             inputs_embeds=inputs_llama_with_s,
#             attention_mask=mask_prefix,
#             max_new_tokens=128,
#             num_beams=5,
#             do_sample=True,
#             min_length=5,
#             top_p=0.9,
#             repetition_penalty=1.0,
#             length_penalty=1,
#             temperature=1.0,
#         )
#         res_text = process_output_tokens(self, temp_res_tokens)
#         return res_text



class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)