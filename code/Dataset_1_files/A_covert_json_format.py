import os
import json
#import pandas as pd
import numpy as np
import pickle

# load ori json
with open('/PATH/TO/MSVD_caption_daxie.json', 'r') as f:
    data = json.load(f)
print(data.keys())  # dict_keys(['metadata', 'train', 'test'])
train_list = data['train']
print("train_list: ", train_list)
test_list = data['test']
print("test_list: ", test_list)
meta_data = data['metadata'] # list
train_info_dict = {}
test_info_dict = {}

train_file = open('./train_dataset1.json', 'a', encoding='utf-8')
test_file = open('./test_dataset1.json', 'a', encoding='utf-8')

#path
train_save_path = '/PATH/TO/Dataset_1_train'
test_save_path = '/PATH/TO/Dataset_1_test'
# train
for train_vid in train_list:
    sub_train_dict = train_info_dict[train_vid] = {}
    for dict_i in meta_data:
        if train_vid == dict_i["video_id"]:
            sub_train_dict["caption"] = dict_i["sentence"]
            sub_train_dict["save_path"] = dict_i["pic_path"]
# test
for test_vid in test_list:
    sub_test_dict = test_info_dict[test_vid] = {}
    for dict_i in meta_data:
        if test_vid == dict_i["video_id"]:
            sub_test_dict["caption"] = dict_i["sentence"]
            sub_test_dict["save_path"] = dict_i["pic_path"]
json_save_train = json.dumps(train_info_dict, ensure_ascii=False)
json_save_test = json.dumps(test_info_dict, ensure_ascii=False)
train_file.write(json_save_train)
test_file.write(json_save_test)
train_file.close()
test_file.close()



