import cv2
import os

def video_to_frames(video_path, output_folder):
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 讀取影片
    cap = cv2.VideoCapture(video_path)
    
    # 檢查影片是否能夠成功打開
    if not cap.isOpened():
        print(f"無法開啟影片檔案: {video_path}")
        return
    
    # 取得影片名稱（無副檔名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        # 如果影片結束，則跳出循環
        if not ret:
            break
        
        # 儲存每一禎的圖片，檔名格式為 [影片名稱_禎數].jpg
        frame_filename = os.path.join(output_folder, f"{video_name}_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    # 釋放影片
    cap.release()
    print(f"影片 {video_path} 完成！總共擷取了 {frame_count} 張圖片。")

def process_all_videos_in_directory(directory):
    # 列出當前資料夾中的所有檔案
    files = os.listdir(directory)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')  # 可根據需要添加更多影片格式
    
    for file_name in files:
        if file_name.lower().endswith(video_extensions):
            video_path = os.path.join(directory, file_name)
            output_folder = os.path.join(directory, os.path.splitext(file_name)[0])
            video_to_frames(video_path, output_folder)

# 使用範例
current_directory = os.getcwd()
process_all_videos_in_directory(current_directory)
