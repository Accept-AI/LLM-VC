import json
import os
from timesformer.models.vit import TimeSformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle

"""
1. read train_video_info.json, to get the videoid list; initialize a Timesformer classifier
2. read PlayerID_bbox_sequences_info.json, to:
    1) format -- videoid: [player1, player2], and get the label of PlayerID, and get the path of each PlayerID;
    2) let the classifier test on Player sequences
3. save the pickle file:
    format -- dict: {videoid: {player1: feature, player2: feature}}
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 路径
player_path = '/PATH/TO/Players'   # all player sequences
# json files
train_video_path = '/PATH/TO/train_video_info.json'
all_bbox_player_video_path = '/PATH/TO/C_PlayerID_bbox_sequences_info.json'
# save pkl -- format dict
E_result_dict = {}
num_frame = 25
image_height, image_width = 224, 224
# 1.
# load all train videoid, to a list
with open(train_video_path, encoding='utf-8') as all_train:
    result_all_train = json.load(all_train)
train_videoid_list = list(result_all_train.keys())
# print(train_videoid_list)  # ['Video100228', 'Video100229', 'Video100230', 'Video100231', 'Video100232', 'Video100233', ..]

# load C_PlayerID_bbox_sequences_info.json
with open(all_bbox_player_video_path, encoding='utf-8') as all_bbox_player_video:
    result_all_bbox_player_video = json.load(all_bbox_player_video)

# initialize a Timesformer classifier
model_path = "/PATH/TO/stage_two/NBA_result_timesformer/[nba]_DFGAR_<2024-09-22_00-19-11>/epoch44_91.13%.pth"
model = TimeSformer(
        img_size=224,
        num_classes=321,
        num_frames=20,
        attention_type="divided_space_time",
        pretrained_model="/PATH/TO/TimeSformer_divST_32x32_224_HowTo100M.pyth",
    )
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
#  pre-process
transformer_player = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# 2.
# 1) format -- videoid: [player1, player2], and get the label of PlayerID, and get the path of each PlayerID;
# 2) let the classifier test on Player sequences
for video_id in train_videoid_list:
    video_player_dict = {}
    for k_P_id, v_dict in result_all_bbox_player_video.items():
        #video_player_dict = {}
        video_id_i = v_dict["Source_ID"]
        player_id_path = v_dict["Sequence_path"]
        player_label = v_dict["Label"]
        if video_id_i == video_id:
            print("video: ", video_id_i)
            try:
                images = []
                fileList = os.listdir(player_id_path)
                new_sort = sorted(fileList, key=lambda i: int((i.split('.')[0]).split('_')[1]))
                #print("images: ", new_sort)
                # if len(new_sort) == 0:
                #     continue
                for i_image in new_sort:
                    #print("i_image: ", i_image)
                    i_path = os.path.join(player_id_path, i_image)
                    #print("i_path: ", i_path)
                    img = Image.open(i_path)
                    img = transformer_player(img)
                    images.append(img)
                images = torch.stack(images)
                if num_frame < images.shape[0]:  # 20 阈值
                    images = images[:num_frame]
                i_l, _, _, _ = images.shape
                padded_video = np.zeros((num_frame, 3, image_height, image_width))
                padded_video[:i_l, :, :, :] = images
                # print("images: ", images.shape)  #images:  torch.Size([X, 3, 360, 640])
                # print("padded_video: ", padded_video.shape) # padded_video:  torch.Size([20, 3, 360, 640])
                video_mask = np.zeros((1, num_frame), dtype=np.compat.long)
                video_mask[0][:i_l] = [1] * i_l
                videos = torch.tensor(padded_video).float().cuda().unsqueeze(0)
                # print("videos.shape: ", videos.shape)
                # 5. analyse the classifier results and record the results in pkl file
                with torch.no_grad():
                    score_i, feature_i = model(videos)  # torch.Size([B, 321])
                    #print("feature_i: ", feature_i.shape)
                video_player_dict[player_label] = feature_i.cpu().numpy()
                E_result_dict[video_id] = video_player_dict
            except:
                pass


print(E_result_dict)
with open("D_train_videoid_player_feature_top2.pkl", "wb") as pkl_file:
    pickle.dump(E_result_dict, pkl_file)

print("完成E_classifier_on_cropped_players")


