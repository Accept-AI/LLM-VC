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
# 让训练好的Timesformer对序列进行分类
1. Timesformer on cropped players
2. record the classifier results, and save top-3 players' features and names
3. save above results (top-3 players for each video_id) in pkl file

"""
#######
#######
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 路径
data_path = '/home/xzy/xzy_nba/VG_NBA_save_player'
# save pkl
E_result_dict = {}

# args
num_frame = 25
image_height, image_width = 224, 224
# 1. 获取球员真值列表
with open("/home/xzy/xzy_nba/LLM_VC/Player_identify/Save/D_all_name_label.json", encoding='utf-8') as Player_list:
    result_player_list = json.load(Player_list)
PLAYERS = result_player_list['all']
name_to_id = {name: i for i, name in enumerate(PLAYERS)}
print("name_to_id: ", name_to_id)   # {"name":"id"}
id_to_name = {str(value): key for key, value in name_to_id.items()}
print("id_to_name: ", id_to_name)   # {"id":"name"}
# 2. Timesformer初始化
model_path = "/home/xzy/xzy_nba/LLM_VC/Player_identify/stage_two/NBA_result_timesformer/[nba]_DFGAR_<2024-09-22_00-19-11>/epoch44_91.13%.pth"
model = TimeSformer(
        img_size=224,
        num_classes=321,
        num_frames=20,
        attention_type="divided_space_time",
        pretrained_model="/home/xzy/xzy_nba/LLM_VC/Player_identify/stage_one/network/TimeSformer/pretrained_model/TimeSformer_divST_32x32_224_HowTo100M.pyth",
    )
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 3. load player files and pre-process
transformer_player = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
with open("/home/xzy/xzy_nba/LLM_VC/Player_identify/code/A_data_preprocessing/D_bbox_for_player_video.json", encoding='utf-8') as video_player:
    result_video_player = json.load(video_player)
    print("number of videos: ", len(result_video_player))   # 1060
jishu = 0
for k_video_id, v_players_info in result_video_player.items():
    jishu += 1
    # 初始化三个列表
    name_list = []
    feature_list = []
    score_list = []
    print("个数：", jishu)
    print("k_video_id: ", k_video_id)
    #print("v_players_info: ", v_players_info)
    num_players = len(v_players_info)   # 每个视频的数量是不一样的
    for player_idi in v_players_info:
        images = []
        path = os.path.join(data_path, k_video_id, player_idi)
        # # 获取该目录下所有文件，存入列表中
        fileList = os.listdir(path)
        new_sort = sorted(fileList, key=lambda i: int((i.split('.')[0]).split('_')[1]))
        #print("new_sort: ", new_sort)
        if len(new_sort) == 0:
            continue
        for i_image in new_sort:
            #print("i_image: ", i_image)
            i_path = os.path.join(path, i_image)
            #print("i_path: ", i_path)
            img = Image.open(i_path)
            img = transformer_player(img)
            images.append(img)

        images = torch.stack(images)
        #print("images: ", images.shape)
        if num_frame < images.shape[0]:  # 20 阈值
            images = images[:num_frame]
        i_l, _, _, _ = images.shape
        padded_video = np.zeros((num_frame, 3, image_height, image_width))
        padded_video[:i_l, :, :, :] = images
        #print("images: ", images.shape)  #images:  torch.Size([X, 3, 360, 640])
        #print("padded_video: ", padded_video.shape) # padded_video:  torch.Size([20, 3, 360, 640])
        video_mask = np.zeros((1, num_frame), dtype=np.compat.long)
        video_mask[0][:i_l] = [1] * i_l
        videos = torch.tensor(padded_video).float().cuda().unsqueeze(0)
        #print("videos.shape: ", videos.shape)
        # 5. analyse the classifier results and record the results in pkl file
        with torch.no_grad():
            score_i, feature_i = model(videos)   # torch.Size([B, 321])
        #print("score: ", score_i)
        index = torch.argmax(score_i, dim=1)
        score_num_i = torch.max(score_i)
        #print("score_num_i: ", score_num_i)
        score_list.append(score_num_i)
        #print("score_num_i: ", score_num_i.item())

        #print("index: ", index)
        name_i = id_to_name[str(int(index[0]))]
        name_list.append(name_i)
        #print("name_i: ", name_i)
        #feature_i = feature_i.cpu().numpy()
        #print("feature_i: ", feature_i)  # 1*768

        feature_list.append(feature_i.cpu().numpy())

    #print("score_list: ", score_list)
    # 提取数值和索引
    score_with_indices = [(tensor.item(), idx) for idx, tensor in enumerate(score_list)]
    # 找到top-3个最大值以及索引
    top3 = sorted(score_with_indices, key=lambda x: x[0], reverse=True)[:2]  # 2: top-2   3: top-3
    #print("top3: ", top3)
    E_sub_dict = E_result_dict[k_video_id] = {}
    for value_top3, index_top3 in top3:
        E_sub_dict[name_list[index_top3]] = feature_list[index_top3]
    #print("E_sub_dict: ", E_sub_dict)

# Save pkl
with open("E_videoid_top2_full_feature.pkl", "wb") as pkl_file:
    pickle.dump(E_result_dict, pkl_file)

print("完成E_classifier_on_cropped_players")