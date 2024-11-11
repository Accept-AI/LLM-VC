import os
import json
import random
import time
"""
anno:
player, action
action: foul, rebound, free throw, turnover, jump ball, shot
save:
{Playerxxx:action}
"""
save_path = "/home/xzy/xzy_nba/LLM_VC/Player_identify/Save/"
with open("/home/xzy/xzy_nba/LLM_VC/Player_identify/Save/C_PlayerID_bbox_sequences_info.json", encoding='utf-8') as Player_info:
    result_p_info = json.load(Player_info)

with open("/home/xzy/xzy_nba/LLM_VC/Player_identify/Save/D_train.json", encoding='utf-8') as train_info:
    result_train = json.load(train_info)
    print("train_info: ", len(result_train))

with open("/home/xzy/xzy_nba/LLM_VC/Player_identify/Save/D_test.json", encoding='utf-8') as test_info:
    result_test = json.load(test_info)
    print("test_info: ", len(result_test))

player5_dict_train = {}
player5_dict_test = {}

# train
for k_train, v_train in result_train.items():
    key = list(result_p_info[k_train]["Source_info"].keys())[0]
    caption = result_p_info[k_train]["Source_info"][key]["Caption"]
    if 'foul' in caption:
        action_label = "Foul"
        player5_dict_train[k_train] = action_label
    if 'rebound' in caption:
        action_label = "Rebound"
        player5_dict_train[k_train] = action_label
    if 'free throw' in caption:
        action_label = "Free Throw"
        player5_dict_train[k_train] = action_label
    if 'Turnover' in caption:
        action_label = "Turnover"
        player5_dict_train[k_train] = action_label
    if 'Jump ball' in caption:
        action_label = "Jump Ball"
        player5_dict_train[k_train] = action_label
    if '-pt' in caption:
        action_label = "Shot"
        player5_dict_train[k_train] = action_label
print("train_action: ", len(player5_dict_train))
train_json_file = open(save_path + 'E_action_train.json', mode='a', encoding='utf-8')

json_save = json.dumps(player5_dict_train, ensure_ascii=False)
train_json_file.write(json_save)
train_json_file.close()
print("train_action_dict.json 保存！")

# train
for k_test, v_test in result_test.items():
    key = list(result_p_info[k_test]["Source_info"].keys())[0]
    caption = result_p_info[k_test]["Source_info"][key]["Caption"]
    if 'foul' in caption:
        action_label = "Foul"
        player5_dict_test[k_test] = action_label
    if 'rebound' in caption:
        action_label = "Rebound"
        player5_dict_test[k_test] = action_label
    if 'free throw' in caption:
        action_label = "Free Throw"
        player5_dict_test[k_test] = action_label
    if 'Turnover' in caption:
        action_label = "Turnover"
        player5_dict_test[k_test] = action_label
    if 'Jump ball' in caption:
        action_label = "Jump Ball"
        player5_dict_test[k_test] = action_label
    if '-pt' in caption:
        action_label = "Shot"
        player5_dict_test[k_test] = action_label
print("test_action: ", len(player5_dict_test))
test_json_file = open(save_path + 'E_action_test.json', mode='a', encoding='utf-8')

json_save = json.dumps(player5_dict_test, ensure_ascii=False)
test_json_file.write(json_save)
test_json_file.close()
print("test_dict.json 保存！")