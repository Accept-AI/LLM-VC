import os
import json
import random
import time

"""
1. select the players more than 5
2. make the json file to record the player file whose number more than 5
3. divide the selected players to training set and testing set
    3.1 collect the player sequences of same name in one sub_dict
    3.2 divide the player sequences adhere to the principle of 8:2
4. make the train 
"""
save_path = '/home/xzy/xzy/caption/LLM_VC/Player_identify/Save/'
# function preparation
def split_list(names_list, ratio=0.8):
    # shuffle the list
    random.shuffle(names_list)

    # calculate the split point
    split_point = int(len(names_list) * ratio)

    # split the training set and testing set
    list1 = names_list[:split_point]
    list2 = names_list[split_point:]

    return list1, list2


# 1.select the players more than 5, and save in a list
player5_list = []
playerless_5 = []
with open("/Player_identify/Save/B_Player_statistic.json", encoding='utf-8') as Player_num:
    result_p_num = json.load(Player_num)
for k_p, v_p in result_p_num.items():
    if int(v_p) > 5 or int(v_p) == 5:
        player5_list.append(k_p)
    if int(v_p) < 5:
        playerless_5.append(k_p)

print("player5_list: ", len(player5_list))  # 321  specific name
print("playerless_5: ", len(playerless_5))  # 37   specific name

# 2. make the json file to record the player file whose number more than 5
player5_dict = {}
with open("/Player_identify/Save/C_PlayerID_bbox_sequences_info.json", encoding='utf-8') as Player_info:
    result_p_info = json.load(Player_info)
for k_info, v_info in result_p_info.items():
    if v_info['Label'] in player5_list:
        player5_dict[k_info] = v_info   # playerxxxx : info
print("player5_dict: ", len(player5_dict))  # player5_dict:  12256

# 3. divide the selected players to training set and testing set
# 3.1 collect the player sequences of same name in one sub_dict
same_player5 = {}
for k_p5, v_p5 in player5_dict.items():
    if v_p5["Label"] not in same_player5:
        same_player5[v_p5["Label"]] = []
        same_player5[v_p5["Label"]].append(k_p5)
    else:
        same_player5[v_p5["Label"]].append(k_p5)
print("same_player5: ", len(same_player5))  # 321

# 3.2 divide the player sequences adhere to the principle of 8:2
train_list = []
test_list = []
for k_s5, v_s5 in same_player5.items():
    #print("ori_list: ", len(v_s5))
    train_sub_list, test_sub_list = split_list(v_s5)
    #print("train_sub_list: ", len(train_sub_list))
    #print("test_sub_list: ", len(test_sub_list))
    #print("-----------")
    train_list = train_list + train_sub_list
    test_list = test_list + test_sub_list
#print("training set: ", train_list)
#print("testing set: ", test_list)
print("training set: ", len(train_list))  # 9686-1-1
print("testing set: ", len(test_list)) # 2570

# 4. make the train and test files
train_dict = {}
test_dict = {}
with open("/Player_identify/Save/C_PlayerID_bbox_sequences_info.json", encoding='utf-8') as Player_info:
    result_p_info = json.load(Player_info)
#i = 0
for train_player in train_list:
    train_label = result_p_info[train_player]["Label"]
    # print(t_player)
    # print(label)
    # if train_player in train_dict:
    #     i = i + 1
    #     print(i)
    #     print("train_t_player", train_player)
    train_dict[train_player] = train_label
    #time.sleep(2)
print("train_dict: ", len(train_dict))
train_json_file = open(save_path + 'D_train.json', mode='a', encoding='utf-8')

json_save = json.dumps(train_dict, ensure_ascii=False)
train_json_file.write(json_save)
train_json_file.close()
print("train_dict.json 保存！")
# print(train_dict)

for t_player in test_list:
    label = result_p_info[t_player]["Label"]
    # print(t_player)
    # print(label)
    if t_player in test_dict:
        print("test_t_player", t_player)
    test_dict[t_player] = label
    #time.sleep(2)
print("test_dict: ", len(test_dict))
test_json_file = open(save_path + 'D_test.json', mode='a', encoding='utf-8')

json_save = json.dumps(test_dict, ensure_ascii=False)
test_json_file.write(json_save)
test_json_file.close()
print("test_dict.json 保存！")

#print("all: ", player5_list)  # as the truth label
truth_label_dict = {}
truth_label_dict["all"] = player5_list
all_name_label_file = open(save_path + 'D_all_name_label.json', mode='a', encoding='utf-8')

json_save = json.dumps(truth_label_dict, ensure_ascii=False)
all_name_label_file.write(json_save)
all_name_label_file.close()
print("D_all_name_label.json 保存！")







