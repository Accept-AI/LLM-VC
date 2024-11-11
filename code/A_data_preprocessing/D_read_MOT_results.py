import json
import os
import cv2
import moviepy.editor as mp
import shutil
"""
1.解析txt中的跟踪结果，保存json,记录相关内容

2.根据json的内容，对每个视频进行裁减--然后将此视频的结果保存到json中

3.json文件记录: 每个视频的bbox for each player in each video  

文件示例：VG_NBA_MOT_player
            Video100000
                1
                    000000.jpg
                    000001.jpg
                    .
                    .
                    .
                2    
                    000000.jpg
"""
# read_txt_file
def read_txt_file(path):
    result_dict = {}
    with open(path) as f:
        for line in f.readlines():
            values = line.split(',')
            #print(values)
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
                #print("frame_id: ", frame_id)
                result_sub = result_dict[frame_id] = {}
                result_sub[player_id] = list(map(int, [x1, y1, x2, y2]))
            else:
                result_dict[frame_id][player_id] = list(map(int, [x1, y1, x2, y2]))

            if values == [''] or values == [' ']:
                # print(sid)
                continue
    #print("result_dict: ", result_dict)
    return result_dict

# 解析函数，分析球员在各帧的情况
def process_player_coordinates(json_data):
    """
    解析字典数据，返回每个球员在所有帧中的坐标列表。
    如果球员在某一帧中没有出现，使用[0,0,0,0]表示。
    参数：
    -data: 包含帧id、球员id和坐标
    返回：
    一个字典，键是球员id，值是包含每一帧坐标的列表
    """
    result = {}
    # 获取所有球员的id
    all_players_list = []
    for k_frame, v_frame_players in json_data.items():
        for k_player_id, v_bbox in v_frame_players.items():
            if k_player_id not in all_players_list:
                all_players_list.append(k_player_id)
            else:
                pass
    #all_player_ids = set(player_id for frame_data in json_data.values() for player_id in frame_data.keys())
    #print("all_players_list: ", all_players_list)

    # 获取所有的帧id，并按顺序遍历
    all_frame_ids = json_data.keys()
    #print("all_frame_ids: ", all_frame_ids)

    for player_i in all_players_list:
        result[player_i] = []

    for frame_i in all_frame_ids:
        for player_id in all_players_list:
            if player_id in json_data[frame_i]:
                result[player_id].append(json_data[frame_i][player_id])
            else:
                result[player_id].append([0, 0, 0, 0])
    #print("process_txt_result: ", result)
    return result


def crop_players(input_folder, coordinates, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, coord in enumerate(coordinates):
        #print(coord)
        x1, y1, x2, y2 = coord
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue
        if x1 < 0 or y1 < 0 or x2 < 0 or y2< 0:
            continue
        #print("x1, y1, x2, y2: ", x1, y1, x2, y2)

        frame_name = f"{idx:06d}.jpg"
        frame_path = os.path.join(input_folder, frame_name)
        #print(frame_path)
        if not os.path.exists(frame_path):
            #print(frame_path)
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            #print("ddddddddd")
            continue

        cropped_player = frame[y1:y2, x1:x2]
        #print("cropped_player: ", cropped_player)
        cropped_frame_name = f"player_{idx:06d}.jpg"
        #print(os.path.join(output_folder, cropped_frame_name))
        #print(input_folder)
        cv2.imwrite(os.path.join(output_folder, cropped_frame_name), cropped_player)

# prepare
# New json, named D_bbox_for_player_video
bbox_dict = {}
# New json file
# 创建json
new_file = open('D_bbox_for_player_video.json', mode='a', encoding='utf-8')

# save_path: to save the cropped players
save_path = "/PATH/TO/VG_NBA_save_player"
# 读取文件夹下所有子文件的txt文件，以及获取路径
# 遍历当前文件夹中的所有子文件夹
#candidate_path = "/PATH/TO/VG_NBA_player_lizi"  # 例子，随便找几个测试下
candidate_path = "/PATH/TO/VG_NBA_data" 

for dirpath, dirnames, filenames in os.walk(candidate_path):
    #print("dirpath: ", dirpath)  
    #print("dirnames: ", dirnames)  # ['Video100009', 'Video100008', 'Video100013', 'Video100004'..]
    #print("filenames: ", filenames)   # []
    for video_id in dirnames:
        print("----------------------------------------------------")
        print("video_id: ", video_id)
        load_MOT_path = os.path.join(dirpath, video_id, "MOT.txt")
        print("load_MOT_path: ", load_MOT_path)

        # 读取txt结果文件，得到初始的字典
        txt_result_dict = read_txt_file(load_MOT_path)

        # 进一步解析结果文件
        player_in_frame_dict = process_player_coordinates(txt_result_dict)
        input_video_path = os.path.join(candidate_path, video_id)
        print("input_video_path: ", input_video_path)

        # crop players
        for k_player_id, v_bboxes in player_in_frame_dict.items():
            #print("k_player_id: ", k_player_id)
            output_player_path = os.path.join(save_path, video_id, k_player_id)
            print("output_player_path: ", output_player_path)
            if not os.path.exists(output_player_path):
                os.makedirs(output_player_path)
            crop_players(input_video_path, v_bboxes, output_player_path)

        # save sub_dict
        bbox_dict[video_id] = player_in_frame_dict

json_save = json.dumps(bbox_dict, ensure_ascii=False)
new_file.write(json_save)
new_file.close()
print("Save!!!")





