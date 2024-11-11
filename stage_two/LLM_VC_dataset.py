import os
import random
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from transformers import AutoTokenizer, LlamaForCausalLM
import copy
import pickle

from build.lib.timesformer.datasets.cv2_transform import pad_image

IGNORE_INDEX = -100
def pad_tensor(tensor, target_shape=(8,3072)):
    current_shape = tensor.shape
    if current_shape [0] < target_shape[0]:
        padding_rows = target_shape[0] - current_shape[0]
        padding_tensor = torch.zeros((padding_rows, target_shape[1]), dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding_tensor], dim=0)
    return tensor

class LLM_dataset(Dataset):
    def __init__(self, feature_root, video_info_path, videoid_top2_path, max_frames, max_words,
                 tokenizer_name='meta-llama/Meta-Llama-3-1B'):

        self.video_info = video_info_path
        self.feature_root = feature_root
        self.train_videoid_top2 = videoid_top2_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.predict_model = LlamaForCausalLM.from_pretrained(tokenizer_name, torch_dtype=torch.bfloat16)
        self.max_frames = max_frames
        self.max_words = max_words

        # load files
        #self.video_feature = os.listdir(feature_root)  # npy files --dim 768
        self.video_info_dict = json.load(open(self.video_info, 'r'))  # "Video100228": {"source_path": "/home/xzy/xzy_nba/VG_NBA_2024/20221111-Phoenix Suns-Orlando Magic/20", "caption": "C.Payne makes 2-pt jump shot from 18 ft", "save_path": "/home/xzy/xzy_nba/VG_NBA_videos_train/Video100228", "game_id": "20221111-Phoenix Suns-Orlando Magic"}
        self.video_top2_pkl = pickle.load(open(self.train_videoid_top2, 'rb'))    # video102694 : {"xxx": array([[xxx]])}
        self.video_feature = list(self.video_top2_pkl.keys())

        self.tokenizer.pad_token_id = 128001
        #self.tokenizer.add_tokens(["[PLAYER]", "[TEAM]", "[COACH]", "[REFEREE]", "([TEAM])"], special_tokens=True)


    def __len__(self):
        #print(len(self.video_feature))
        return len(self.video_feature)

    def __getitem__(self, index):

        # video
        video_id = self.video_feature[index]  # get the video_id

        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        video = np.zeros((1, self.max_frames, 768), dtype=np.float32)
        #print("video: ", video.shape)
        video_feature = torch.from_numpy(np.load(os.path.join(self.feature_root, video_id, 'out.npy')))
        if video_feature.shape[0] > self.max_frames:
            #print("video_feature:", video_feature.shape[0])
            video_index = video_feature.shape[0] - self.max_frames
            #print("video_index:", video_index)
            video[0] = video_feature[video_index:]
            video_mask[0][:self.max_frames] = [1] * self.max_frames

        else:    # self.max_frames >= video_feature.shape[0]:  # max_frame 阈值
            video_mask[0][:video_feature.shape[0]] = [1] * video_feature.shape[0]
            video[0][:video_feature.shape[0]] = video_feature

        # entity prompt
        with torch.no_grad():

            entity_dict = self.video_top2_pkl[video_id]
            entity_list = list(entity_dict.keys())
            entity_one = entity_list[0]
            entity_one_feature = entity_dict[entity_one]
            name_one_tokens = self.tokenizer(
                text=entity_one,
                return_tensors="pt",
                max_length=128,
                truncation=True
            ).input_ids[0]
            name_one_embeds = self.predict_model.model.embed_tokens(name_one_tokens)
            name_one_embeds = name_one_embeds[1:]
            name_one = pad_tensor(name_one_embeds)
            #print("name_one: ", name_one.shape)
            #name_one_embeds = torch.mean(name_one_embeds, dim=0)
            try:
                entity_two = entity_list[1]
                entity_two_feature = entity_dict[entity_two]
                name_two_tokens = self.tokenizer(
                    text=entity_two,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                ).input_ids[0]
                name_two_embeds = self.predict_model.model.embed_tokens(name_two_tokens)
                name_two_embeds = name_two_embeds[1:]
                name_two = pad_tensor(name_two_embeds)
                #print("name_two: ", name_two.shape)

                #name_two_embeds = torch.mean(name_two_embeds, dim=0)
            except:
                name_two_tokens = name_one_tokens
                #name_two_embeds = name_one_embeds
                name_two = name_one
                #print("name_two: ", name_two.shape)

                entity_two_feature = entity_one_feature


        # caption
        caption = self.video_info_dict[video_id]["caption"] + "<|end_of_text|>"
        caption_tokens = self.tokenizer(
            text=caption,
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).input_ids[0]
        labels = copy.deepcopy(caption_tokens)

        # 补0直到长度达到了30
        while len(caption_tokens) < 30:  # 补0直到长度达到了30
            caption_tokens = torch.cat((
                caption_tokens,
                torch.tensor([self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")])))

            labels = torch.cat((
                labels,
                torch.tensor([-100])))

        attention_mask = caption_tokens.ne(self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"))

        return video, video_mask, caption_tokens.detach(), labels.detach(), attention_mask.detach(), name_one.detach(), entity_one_feature, name_two.clone().detach(), entity_two_feature, self.video_info_dict[video_id]["caption"]



