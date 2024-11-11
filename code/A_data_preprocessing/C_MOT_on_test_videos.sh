#!/bin/bash

# JSON 文件路径
json_file="/PATH/TO/test_video_info.json"

# 使用 Python 来解析 JSON，并返回每个 video_id 和 save_path
python3 - <<END | while IFS=' ' read -r video_id video_path; do
import json

# 打开并读取 JSON 文件
json_file_path = "$json_file"
with open(json_file_path, 'r', encoding='utf-8') as f:
    video_info = json.load(f)

# 打印出 video_id 和 save_path，作为后续 Shell 脚本处理的输出
for video_id, video_data in video_info.items():
    save_path = video_data['save_path']
    print(f"{video_id} {save_path}")
END

# 在 Shell 中逐行读取 Python 输出，并将其作为参数传递给 script.py
    # 调用 Python 脚本，并传入 video_id 和 save_path 作为参数
    python3 C_MOT_on_test_videos.py --video_id "$video_id" --video_path "$video_path"

done

