import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
import os

import numpy as np
import random
from PIL import Image

ACTIVITIES = ['2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
              '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.']

with open("/home/xzy/xzy_nba/LLM_VC/Player_identify/Save/D_all_name_label.json", encoding='utf-8') as Player_list:
    result_player_list = json.load(Player_list)
with open("/home/xzy/xzy_nba/LLM_VC/Player_identify/Save/D_all_action_label.json", encoding='utf-8') as Action_list:
    result_action_list = json.load(Action_list)
PLAYERS = result_player_list['all']
ACTIONS = result_action_list['all']
print("PLAYERS: ", len(PLAYERS))
print("ACTIONS: ", len(ACTIONS))

def read_ids(path):
    with open(path, encoding='utf-8') as data_f:
        result_data = json.load(data_f)
    data_list = list(result_data.keys())
    return data_list

def nba_read_annotations(player_path, action_path, seqs):
    labels = {}
    name_to_id = {name: i for i, name in enumerate(PLAYERS)}
    action_to_id = {action: j for j, action in enumerate(ACTIONS)}
    with open(player_path, encoding='utf-8') as player_data_f:
        result_player_data = json.load(player_data_f)

    with open(action_path, encoding='utf-8') as action_data_f:
        result_action_data = json.load(action_data_f)

    for sid in seqs:
        annotations = {}
        player_label = result_player_data[sid]
        player_id = name_to_id[player_label]

        action_label = result_action_data[sid]
        action_id = action_to_id[action_label]
        annotations[sid] = {
            'player_id': player_id,
            'action_id': action_id
        }
        labels[sid] = annotations

    return labels



def nba_all_frames(labels):
    frames = []
    for sid, anns in labels.items():
        for fid, ann in anns.items():
            frames.append((sid, ann))
    #print("frames: ", frames)
    return frames


class NBADataset(data.Dataset):
    """
    Volleyball Dataset for PyTorch
    """
    def __init__(self, frames, anns, image_path, args):
        super(NBADataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.image_width = args.image_width
        self.image_height = args.image_height
        self.num_frame = args.num_frame

        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        #frames = self.select_frames(self.frames[idx])
        samples = self.load_samples(self.frames[idx])

        return samples

    def __len__(self):
        return len(self.frames)

    def load_samples(self, frames):
        images, players = [], []
        #print("frames: ", frames)
        vid, sid = frames
        #print("vid: ", vid)
        #print("sid: ", sid)
        path = os.path.join(self.image_path, vid)
        # 获取该目录下所有文件，存入列表中
        fileList = os.listdir(path)
        # get_key是sorted函数用来比较的元素，该处用lambda表达式替代函数。
        #get_key = lambda i: int(i.split('.')[0])
        new_sort = sorted(fileList, key=lambda i: int((i.split('.')[0]).split('_')[1]))
        for i_image in new_sort:
            i_path = os.path.join(self.image_path, vid, i_image)
            #print("i_path: ", i_path)
            img = Image.open(i_path)
            img = self.transform(img)
            images.append(img)
            #players.append(sid['player_id'])

        images = torch.stack(images)
        if self.num_frame < images.shape[0]:  # 20 阈值
            images = images[:self.num_frame]
        i_l, _, _, _ = images.shape
        padded_video = np.zeros((self.num_frame, 3, self.image_height, self.image_width))
        padded_video[:i_l, :, :, :] = images
        #print("images: ", images.shape)  #images:  torch.Size([X, 3, 360, 640])
        #print("padded_video: ", padded_video.shape) # padded_video:  torch.Size([20, 3, 360, 640])
        video_mask = np.zeros((1, self.num_frame), dtype=np.long)
        video_mask[0][:i_l] = [1] * i_l
        #print("video_mask: ", video_mask.shape) # video_mask:  (1, 20)

        players = [sid['player_id']] * self.num_frame
        actions = [sid['action_id']] * self.num_frame

        players = np.array(players, dtype=np.int32)
        actions = np.array(actions, dtype=np.int32)
        #print("players: ", players.shape)  # players:  (20,)
        #print("actions: ", actions.shape)  # actions:  (20,)


        # convert to pytorch tensor
        players = torch.from_numpy(players).long()
        actions = torch.from_numpy(actions).long()


        return padded_video, video_mask, players, actions
