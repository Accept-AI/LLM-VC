import os
import json
import matplotlib.pyplot as plt
"""
Player statistic
make a json file to record the number of player
"""
file_dict = {}
x_list = []
y_list = []

file_player = {}



player_dict = {}
save_path = '/PATH/TO/Save/'
new_file = open(save_path + 'B_Player_statistic.json', mode='a', encoding='utf-8')

#
with open("/PATH/TO/A_VideoID_path.json", encoding='utf-8') as VideoID_f:
    result_vid = json.load(VideoID_f)
# print(len(result_vid))    # 9939
for k_vid, v_vid in result_vid.items():
    a = v_vid
    num = a.split("/")[-1]
    file = num + ".json"
    file_path = os.path.join(v_vid, file)
    with open(file_path, encoding='utf-8') as file_f:
        #print(file_path)
        result_f = json.load(file_f)
        for r_k, r_v in result_f.items():
            r_bbox = r_v["bbox"]
            for b_k, b_v in r_bbox.items():

                # if b_k == "D.Russellmakes":
                #     print(v_vid)

                if b_k not in file_dict:
                    file_dict[b_k] = []
                    file_dict[b_k].append("*")
                else:
                    file_dict[b_k].append("*")

print("字典长度：", len(file_dict))
for f_k, f_v in file_dict.items():
    x_list.append(f_k)
    y_list.append(len(f_v))
    file_player[f_k] = len(f_v)
    print(f_k)
    print(len(f_v))
    player_dict[f_k] = len(f_v)
    print("------------")

# save
json_save = json.dumps(player_dict, ensure_ascii=False)
new_file.write(json_save)
new_file.close()
print("Save！")




