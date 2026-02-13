import os
import cv2
import shutil
import numpy as np

import os
import cv2
import shutil
import numpy as np

def cut_video(video, img_save_path):
    global index
    crop_frames = 0
    path_sample = '/home/nas/Research_Group/Personal/Johnny/0000.jpg'
    cap = cv2.VideoCapture(video)
    shape = cv2.imread(path_sample).shape

    src = np.float32(ROIs)
    h, w = shape[0], shape[1]
    dst = np.float32([(0, 0), (0, h), (w, h), (w, 0)])
    M = cv2.getPerspectiveTransform(src, dst)

    video_name = video.split('/')[-1]  # Changed '\\' to '/'
    pa = video_name.split('.')[-2]
    p = os.path.join(img_save_path, pa)  # Changed concatenation to os.path.join for correctness

    try:
        os.makedirs(p)
    except FileExistsError:
        print(p + " exists")
    oldpath = os.path.join(p, "images")  # Changed concatenation to os.path.join for correctness

    try:
        os.makedirs(oldpath)
    except FileExistsError:
        print(oldpath + " exists")

    target_image = None
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        clone = frame.copy()
        processed = cv2.warpPerspective(clone, M, (w, h))

        if target_image is None:
            n = filename_n - len(str(crop_frames + index))
            filename = str(0) * n + str(crop_frames)
            save = os.path.join(oldpath, '{}.jpg'.format(filename))  # Changed concatenation to os.path.join for correctness
            cv2.imwrite(save, processed)
            target_image = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            crop_frames += 1
            continue

        s = 1  # This should probably be a comparison between target_image and the current processed frame
        if s <= similarity:
            n = filename_n - len(str(crop_frames + index))
            filename = str(0) * n + str(crop_frames + index)
            save = os.path.join(oldpath, '{}.jpg'.format(filename))  # Changed concatenation to os.path.join for correctness
            cv2.imwrite(save, processed)
            crop_frames += 1
            target_image = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

# 主程序
root_path = '/home/nas/Research_Group/Personal/Johnny/data_batch_backup'
pen = 'fourth'
batch = '20230622-0730'
dayy = '20230708'
video_folder_path = os.path.join(root_path, pen, batch, dayy, 'videos')
filename_n = 4              # 檔名字數(6:前面補0至6位數 000001)
sep = True                  # 1分鐘影片切成30秒
vdo = False

# ROIs = [(142, 3), (157, 542), (959, 541), (958, 2)]  ## second pen
ROIs = [(26, 44), (46, 541), (857, 508), (801, 5)]  ## forth pen 
# ROIs = [(188, 3), (190, 542), (959, 543), (959, 8)]  ## third pen

similarity = 1
index = 0
video_path = ''  # Initialize an empty string for video_path
for time in os.listdir(video_folder_path):
    folder_time = os.path.join(video_folder_path, time)
    for video in os.listdir(folder_time):
        if video.endswith('.avi'):
            video_path = os.path.join(folder_time, video)
            print("Video path: ", video_path)
            img_save_path = os.path.join(root_path, pen, batch, dayy, 'cropped2')  # 校正後影像位置
            cut_video(video_path, img_save_path)  # Moved cut_video call into the loop