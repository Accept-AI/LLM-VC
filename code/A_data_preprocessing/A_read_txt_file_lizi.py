import json
import os

Out_save_folder = '/PATH/TO/A_data_preprocessing/'

# Out json
result_dict = {}

# 创建json
new_file = open(Out_save_folder + 'result_detail.json', mode='a', encoding='utf-8')


path = '/PATH/TO/A_data_preprocessing/'

with open(path + 'MOT_result.txt') as f:
    for line in f.readlines():
        values = line.split(',')
        print(values)
        frame_id = values[0]
        player_id = values[1]
        x1 = values[2]
        y1 = values[3]
        w = values[4]
        h = values[5]
        x2 = float(x1) + float(w)
        y2 = float(y1) + float(h)
        x1 = int(float(x1))
        x2 = int(float(x2))
        y1 = int(float(y1))
        y2 = int(float(y2))

        if frame_id not in result_dict:
            print("frame_id: ", frame_id)
            result_sub = result_dict[frame_id] = {}
            result_sub[player_id] = list(map(int, [x1, y1, x2, y2]))
        else:
            result_dict[frame_id][player_id] = list(map(int, [x1, y1, x2, y2]))

        if values == [''] or values == [' ']:
            # print(sid)
            continue
print(result_dict)

json.dump(result_dict, new_file)
new_file.close()

