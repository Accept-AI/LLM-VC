import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import random
from PIL import Image

ACTIVITIES = ['2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
              '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.']


def read_ids(path):
    file = open(path)
    values = file.readline()
    values = values.split(',')[:-1]
    values = list(map(int, values))

    return values


def nba_read_annotations(path, seqs):
    labels = {}
    group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}

    for sid in seqs:
        annotations = {}
        #print(path)
        #with open(path + '/%d/annotations.txt' % sid) as f:   # 原代码
        with open(path + '/%d/annotation_plus.txt' % sid) as f:
            for line in f.readlines():
                #values = line[:-1].split('\t')    # 原代码
                values = line[:-1].split(' ' * 6)
                # print(values)

                if values == [''] or values == [' ']:
                    # print(sid)
                    continue
                file_name = values[0]
                fid = int(file_name.split('.')[0])

                activity = group_to_id[values[1]]

                annotations[fid] = {
                    'file_name': file_name,
                    'group_activity': activity,
                }
            labels[sid] = annotations

    return labels
def nba_read_test_annotations(path, seqs):
    labels = {}
    group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    #print(group_to_id)

    for sid in seqs:
        annotations = {}
        #print(path)
        with open(path + '/%d/annotations.txt' % sid) as f:
        #with open(path + '/%d/annotation_plus.txt' % sid) as f:
            for line in f.readlines():
                values = line[:-1].split('\t')
                #print(values)


                #print(values[1]) # 标签
                file_name = values[0]

                #print(file_name)
                #print(sid)  # 21801058  102756
                #print(file_name)  # 15
                fid = int(file_name.split('.')[0])
                #fid = int(file_name)
                #print(fid)
                #label_nba = file_name.split('.')[1]
                #print(label_nba)

                activity = group_to_id[values[1]]

                annotations[fid] = {
                    'file_name': file_name,
                    'group_activity': activity,
                    'label': values[1]
                }
            labels[sid] = annotations
    #print(labels)

    return labels

def nba_all_frames(labels):
    frames = []

    for sid, anns in labels.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))

    return frames


class NBADataset(data.Dataset):
    """
    Volleyball Dataset for PyTorch
    """
    def __init__(self, frames, anns, image_path, args, is_training=True):
        super(NBADataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_total_frame = args.num_total_frame
        self.is_training = is_training
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        frames = self.select_frames(self.frames[idx])
        samples = self.load_samples(frames)

        return samples

    def __len__(self):
        return len(self.frames)

    def select_frames(self, frame):
        """
        Select one or more frames
        """
        vid, sid = frame

        if self.is_training:
            if self.random_sampling:
                sample_frames = random.sample(range(72), self.num_frame)
                sample_frames.sort()
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame)
        else:
            if self.num_frame == 6:
                # [6, 18, 30, 42, 54, 66]
                sample_frames = list(range(6, 72, 12))
            elif self.num_frame == 12:
                # [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70]
                sample_frames = list(range(4, 72, 6))
            elif self.num_frame == 18:
                # [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]
                sample_frames = list(range(2, 72, 4))
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + segment_duration // 2

        return [(vid, sid, fid) for fid in sample_frames]

    def load_samples(self, frames):
        images, activities = [], []

        for i, (vid, sid, fid) in enumerate(frames):
            fid = '{0:06d}'.format(fid)
            img = Image.open(self.image_path + '/%d/%d/%s.jpg' % (vid, sid, fid))
            img = self.transform(img)

            images.append(img)
            activities.append(self.anns[vid][sid]['group_activity'])

        images = torch.stack(images)
        activities = np.array(activities, dtype=np.int32)

        # convert to pytorch tensor
        activities = torch.from_numpy(activities).long()

        return images, activities
class NBA2022(data.Dataset):
    """
    Volleyball Dataset for PyTorch
    """
    def __init__(self, frames, anns, image_path, args, is_training=True):
        super(NBA2022, self).__init__()
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame    # 18
        self.num_total_frame = args.num_total_frame    # 72
        self.is_training = is_training   # True
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_clip = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __getitem__(self, idx):
        frames = self.select_frames(self.frames[idx])
        samples = self.load_samples(frames)  # 图像
        #print(samples)
        return samples

    def __len__(self):
        return len(self.frames)

    def select_frames(self, frame):
        """
        Select one or more frames
        """
        vid, sid = frame

        if self.is_training:
            if self.random_sampling:
                sample_frames = random.sample(range(72), self.num_frame)
                sample_frames.sort()
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame)
        else:
            if self.num_frame == 6:
                # [6, 18, 30, 42, 54, 66]
                sample_frames = list(range(6, 72, 12))
            elif self.num_frame == 12:
                # [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70]
                sample_frames = list(range(4, 72, 6))
            elif self.num_frame == 18:
                # [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]
                sample_frames = list(range(2, 72, 4))
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + segment_duration // 2
        #print([(vid, sid, fid)for fid in sample_frames])   # [(21801211, 517, 2), (21801211, 517, 5), (21801211, 517, 9),....]
        #print(sample_frames)
        #print([(vid, sid, fid)for fid in sample_frames]))
        return [(vid, sid, fid) for fid in sample_frames]

    def load_samples(self, frames):
        images, activities, text_nba, image_clip = [], [], [], []

        for i, (vid, sid, fid) in enumerate(frames):
            fid = '{0:06d}'.format(fid)
            img1 = Image.open(self.image_path + '/%d/%d/%s.jpg' % (vid, sid, fid))
            #print(img1.size)
            img = self.transform(img1)
            img_clip = self.transform_clip(img1)
            #print(img.shape)
            img2 = torch.clamp(img_clip, 0.0, 1.0)
            #print(self.anns[vid][sid])  #{'file_name': '91', 'group_activity': 8, 'label': '3p-fail.-def.'}
            images.append(img)
            image_clip.append(img2)

            #image_clip.append(img1)
            activities.append(self.anns[vid][sid]['group_activity'])
            #text_nba.append(self.anns[vid][sid]['label'])

        images = torch.stack(images)
        image_clips = torch.stack(image_clip)
        #image_clip = torch.stack(image_clip)
        activities = np.array(activities, dtype=np.int32)

        # convert to pytorch tensor
        activities = torch.from_numpy(activities).long()

        return images, activities