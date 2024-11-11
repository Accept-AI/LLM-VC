import os
import json
import cv2
import moviepy.editor as mp
import shutil

"""
1. Crop player's bbox
2. record the players' information as the json
for example:
    {"Player100000": 
        {"Source":"/media/xzy/58bff591-e818-4a4b-8c3e-e857128495ed/xzy/VG_NBA_2024/20221107-Cleveland Cavaliers-Los Angeles Clippers/193",
        "Source_ID":"Video100000",
        "GameID":"20221107-Brooklyn Nets-Dallas Mavericks",
        "Source_info":{"20221107-Brooklyn Nets-Dallas Mavericks":{"img_size":[1280,720,3],"img_num":750,"st_time":4,"ed_time":6.8,"Caption":"Jump ball: N.Claxton vs. J.McGee (S.Dinwiddie gains possession)","bbox":{"N.Claxton":[[577,255,656,334],[577,255,654,328],[579,258,661,333],[577,255,655,334],[578,259,658,336],[577,255,656,336],[578,254,655,333],[579,262,658,333],[579,260,654,334],[573,247,662,331],[537,229,671,330],[551,236,660,337],[565,240,661,350],[601,272,660,353],[581,278,648,376]],"J.McGee":[[655,243,728,330],[651,244,727,328],[653,245,725,325],[650,240,726,327],[646,246,724,328],[640,245,717,324],[641,252,713,327],[645,253,714,326],[642,253,719,336],[618,210,740,335],[655,214,729,338],[655,208,743,348],[658,248,738,395],[651,279,713,397],[611,287,684,401]],"S.Dinwiddie":[[990,222,1085,324],[990,227,1086,324],[986,228,1087,326],[983,229,1087,325],[984,228,1083,322],[984,227,1081,324],[984,225,1081,325],[984,225,1078,326],[984,224,1078,325],[984,224,1078,325],[984,224,1078,325],[970,232,1071,339],[968,260,1090,349],[1019,276,1124,355],[1052,292,1142,408]]}}},
        "BBox":[[577,255,656,334],[577,255,654,328],[579,258,661,333],[577,255,655,334],[578,259,658,336],[577,255,656,336],[578,254,655,333],[579,262,658,333],[579,260,654,334],[573,247,662,331],[537,229,671,330],[551,236,660,337],[565,240,661,350],[601,272,660,353],[581,278,648,376]],
        "Label":"N.Claxton",
        "Sequence_path":"../....."},
    "Player100001":{...},...}

"""
# 函数
def extract_video_segment(input_video_path, output_video_path, start_time, end_time):

    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)

    video = mp.VideoFileClip(input_video_path).subclip(start_time, end_time)
    save_path = os.path.join(output_video_path, "out.mp4")
    video.write_videofile(save_path, audio=True, codec="mpeg4")

def extract_frames(video_path, output_folder, fps=5):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_path = os.path.join(video_path, "out.mp4")
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get the original FPS of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps // fps)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"frame_{saved_frame_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()

def crop_players(input_folder, coordinates, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, coord in enumerate(coordinates):
        x1, y1, x2, y2 = coord
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue

        frame_name = f"frame_{idx:05d}.jpg"
        frame_path = os.path.join(input_folder, frame_name)
        if not os.path.exists(frame_path):
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        cropped_player = frame[y1:y2, x1:x2]
        cropped_frame_name = f"player_{idx:05d}.jpg"
        cv2.imwrite(os.path.join(output_folder, cropped_frame_name), cropped_player)



###########################3
# Make PlayerID
start = 100000
end = 300000
PLAYERID = [f"Player{i}" for i in range(start, end + 1)]
i = 0  # 记数

# Out path
Out_save_folder = "/media/xzy/58bff591-e818-4a4b-8c3e-e857128495ed/xzy/Players"
output_video_segment_path = "/media/xzy/58bff591-e818-4a4b-8c3e-e857128495ed/xzy/Players/video_folder"
out_frames_path = "/media/xzy/58bff591-e818-4a4b-8c3e-e857128495ed/xzy/Players/frames"
# Out json
Player_dict = {}
error_dict = {}
# 创建json
new_file = open(Out_save_folder + 'C_PlayerID_bbox_sequences_info.json', mode='a', encoding='utf-8')

error_file = open(Out_save_folder + 'C_Error.json', mode='a', encoding='utf-8')

# obtain each VideoID_info
with open("/Player_identify/Save/A_VideoID_path.json", encoding='utf-8') as VideoID_f:
    result_vid = json.load(VideoID_f)

# iterate through the dictionary  遍历字典
for k_vid, v_vid in result_vid.items():
    a = v_vid
    num = a.split("/")[-1]  # 获取文件夹数字名称

    # obtain json
    file_json = num + ".json"
    json_path = os.path.join(v_vid, file_json)

    # obtain video
    file_video = num + ".mp4"
    video_path = os.path.join(v_vid, file_video)

    # read the json
    with open(json_path, encoding='utf-8') as json_f:
        result_json = json.load(json_f)


    for r_k, r_v in result_json.items():
        gameid = r_k
        BBOX = r_v["bbox"]
        start_time = r_v["st_time"]  # 开始时间
        end_time = r_v["ed_time"]    # 结束时间
        for name, bbox in BBOX.items():
            Player_id = PLAYERID[i]
            cropped_folder = os.path.join(Out_save_folder, Player_id)
            Sub_player_dict = Player_dict[Player_id] = {}
            Sub_player_dict["Source"] = v_vid
            Sub_player_dict["Source_ID"] = k_vid
            Sub_player_dict["GameID"] = gameid
            Sub_player_dict["Source_info"] = result_json
            Sub_player_dict["BBox"] = bbox
            Sub_player_dict["Label"] = name
            Sub_player_dict["Sequence_path"] = cropped_folder
            # out path -- to delete
            output_video_segment_path = os.path.join(output_video_segment_path, k_vid)
            out_frames_path = os.path.join(out_frames_path, k_vid)

            # 开始裁剪
            #print("-----------------------")
            #print("开始裁剪：", Player_id)
            # Extract video segment
            extract_video_segment(video_path, output_video_segment_path, start_time, end_time)

            # Extract frames from the video
            extract_frames(output_video_segment_path, out_frames_path, fps=5)

            # Crop players from the extracted frames
            try:
                crop_players(out_frames_path, bbox, cropped_folder)
            except:
                print("问题文件：", v_vid)
                error_dict[v_vid] = bbox


            try:
                shutil.rmtree(output_video_segment_path)
                #print(f"success remove the folder: {output_video_segment_path}")
            except OSError as e:
                pass
                #print(f"the error in delting the folder: {e.strerror}")

            try:
                shutil.rmtree(out_frames_path)
                #print(f"success remove the folder: {out_frames_path}")
            except OSError as e:
                pass
                #print(f"the error in delting the folder: {e.strerror}")
            i = i + 1
            output_video_segment_path = "/media/xzy/58bff591-e818-4a4b-8c3e-e857128495ed/xzy/Players/video_folder"
            out_frames_path = "/media/xzy/58bff591-e818-4a4b-8c3e-e857128495ed/xzy/Players/frames"
            #print("-----------------------")

json_save = json.dumps(Player_dict, ensure_ascii=False)
new_file.write(json_save)
new_file.close()

error_save = json.dumps(error_dict, ensure_ascii=False)
error_file.write(error_save)
error_file.close()

print("保存！")














