import numpy as np
# llama3.2  Qwen2.5
from transformers import AutoTokenizer,LlamaForCausalLM, Qwen2ForCausalLM, GPT2Model
# gpt2
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

import json
import torch
import pickle
import copy
# /home/xzy/xzy_nba/gpt2
# /home/xzy/xzy_nba/Qwen/Qwen2.5-3B
# /home/xzy/xzy_nba/meta-llama/Llama-3.2-3B
tokenizer_name = '/home/xzy/xzy_nba/gpt2'
llm_ckpt = "/home/xzy/xzy_nba/gpt2"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
config = GPT2Config.from_pretrained(tokenizer_name)
predict_model = GPT2LMHeadModel.from_pretrained(llm_ckpt, config=config, torch_dtype=torch.bfloat16)

# gpt2
# tokenizer_name = '/home/xzy/xzy_nba/gpt2'
# config = GPT2Config.from_pretrained(tokenizer_name, add_cross_attention=True)
# predict_model = GPT2LMHeadModel.from_pretrained(tokenizer_name, config=config)
# tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
# /home/xzy/xzy_nba/LLM_VC/Player_identify/code/B_data_preprocessing/train_video_info.json
# /home/xzy/xzy_nba/LLM_VC/Player_identify/code/B_data_preprocessing/test_video_info.json
# /home/xzy/xzy_nba/LLM_VC/Player_identify/Save/C_PlayerID_bbox_sequences_info.json

word_token_list = []

with open("/home/xzy/xzy_nba/LLM_VC/Player_identify/code/B_data_preprocessing/train_video_info.json", encoding='utf-8') as Video_f:
    result_info = json.load(Video_f)


for k_video_info, v_video_info in result_info.items():
    caption = v_video_info['caption'] + " " + "<|endoftext|>"
    # caption = "<|begin_of_text|>" + caption + "<|end_of_text|>"  # 128000, 128000,..., 128001
    #caption = caption + "<|end_of_text|>"
    caption_tokens = tokenizer(
        text=caption,
        return_tensors="pt",
        max_length=128,
        truncation=True
    ).input_ids[0]
    #print(caption_tokens)
#
    for item in caption_tokens.tolist():
        if item not in word_token_list:
            word_token_list.append(item)


with open("/home/xzy/xzy_nba/LLM_VC/Player_identify/code/B_data_preprocessing/test_video_info.json", encoding='utf-8') as Video_f_test:
    result_info_test = json.load(Video_f_test)

for k_video_info_test, v_video_info_test in result_info_test.items():
    caption = v_video_info_test['caption'] + " " + "<|endoftext|>"
    # caption = "<|begin_of_text|>" + caption + "<|end_of_text|>"  # 128000, 128000,..., 128001
    # caption = caption + "<|end_of_text|>"
    caption_tokens = tokenizer(
        text=caption,
        return_tensors="pt",
        max_length=128,
        truncation=True
    ).input_ids[0]

    for item in caption_tokens.tolist():
        if item not in word_token_list:
            word_token_list.append(item)

with open("/home/xzy/xzy_nba/LLM_VC/Player_identify/code/B_data_preprocessing/" + "C_VG_nba_gpt2.pkl", "wb") as pkl_file:
    pickle.dump(word_token_list, pkl_file)

print("完成")