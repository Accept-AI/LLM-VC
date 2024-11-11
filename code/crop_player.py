import cv2
import moviepy.editor as mp
import os
import json
from tqdm import tqdm

def extract_video_segment(input_video_path, output_video_path, start_time, end_time):
    video = mp.VideoFileClip(input_video_path).subclip(start_time, end_time)
    video.write_videofile(output_video_path, audio=True, codec="mpeg4")

def save_frames(video_path, output_folder, fps):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps // fps)

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()

def crop_players_images(frames_folder, bbox_data, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for player, bboxes in bbox_data.items():
        player_folder = os.path.join(output_folder, player)
        if not os.path.exists(player_folder):
            os.makedirs(player_folder)

        for i, bbox in enumerate(bboxes):
            frame_path = os.path.join(frames_folder, f"frame_{i}.jpg")
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                x1, y1, x2, y2 = bbox
                cropped_img = frame[y1:y2, x1:x2]
                cropped_img_path = os.path.join(player_folder, f"{player}_frame_{i}.jpg")
                cv2.imwrite(cropped_img_path, cropped_img)

# Load the JSON data
with open('/PATH/TO//20221107-Brooklyn Nets-Dallas Mavericks/40/40.json') as f:
    data = json.load(f)

# Example usage
input_video_path = '/PATH/TO/3_LLM_VC\Player identification\samples/20221107-Brooklyn Nets-Dallas Mavericks/40/40.mp4'
output_video_segment_path = '/PATH/TO/3_LLM_VC\Player identification\samples/20221107-Brooklyn Nets-Dallas Mavericks/40\out/output_segment.mp4'
video_data = data["20221107-Brooklyn Nets-Dallas Mavericks"]
start_time = video_data["st_time"]
end_time = video_data["ed_time"]
frames_folder = 'frames'
output_folder = 'cropped_frames'

# Extract video segment
extract_video_segment(input_video_path, output_video_segment_path, start_time, end_time)

# Save frames from video segment
save_frames(output_video_segment_path, frames_folder, fps=5)

# Crop players' images from frames
crop_players_images(frames_folder, video_data["bbox"], output_folder)
