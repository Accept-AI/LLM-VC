#E:\BJUT\work\3_LLM_VC\Player identification\code\frames

import cv2
import moviepy.editor as mp
import os
import json
frame_path = "/PATH/TO/LLM_VC/Player_identify/code/MOT/MixSort/Images/67/000000.jpg"
cropped_img_path = "/PATH/TO/LLM_VC/Player_identify/code/"
cropped_img_path = os.path.join(cropped_img_path, "1.jpg")
frame = cv2.imread(frame_path)
print(frame.shape)
#   261.27,511.05,90.38,113.13,0.96
x1, y1, w, h = 261.27,511.05,90.38,113.13
x2 = 261.27 + 90.38
y2 = 511.05 + 113.13
cropped_img = frame[int(y1):int(y2), int(x1):int(x2)].copy()
cv2.imwrite(cropped_img_path, cropped_img)
cv2.imshow("CropDemo", cropped_img)  # 在窗口显示 彩色随机图像
cv2.waitKey(0)
cv2.destroyAllWindows()
