import os
import json
import pickle



# load all file
vid_folders = os.listdir("/PATH/TO/name_json")
#print(vid_folders)
new_dict = {}
new_file = open('./name_mapping.json', mode='a', encoding='utf-8')
for vid_folder in vid_folders:
    path = os.path.join("/PATH/TO/name_json", vid_folder)
    #print(path)
    json_file = json.load(open(path, 'r'))
    #print(json_file)
    for k_name, v_name in json_file.items():
        if k_name not in new_dict:
            new_dict[k_name] = v_name
print(len(new_dict))  # 522
json_save = json.dumps(new_dict, ensure_ascii=False)
new_file.write(json_save)
new_file.close()

