import os
import numpy as np
import cv2  # OpenCV for video processing
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--videos_path', type=str, required=True)  # 视频文件的根路径
parser.add_argument('--train_list', type=str, required=True)   # kinetics700_train_list_videos.txt 文件路径
parser.add_argument('--labels_file', type=str, required=True)  # kinetics_700_labels.csv 文件路径
parser.add_argument('--save_path', type=str, required=True)    # 保存处理后文件的路径
args = parser.parse_args()

# 创建保存文件夹
os.makedirs(args.save_path, exist_ok=True)

# 读取标签文件
labels_df = pd.read_csv(args.labels_file)
labels_dict = labels_df.set_index('id').to_dict()['name']  # 创建 id -> 动作标签的映射

# 读取 train_list 文件
with open(args.train_list, 'r') as f:
    video_list = f.readlines()

# 遍历每个视频文件，并根据 ID 获取对应的动作
for line in tqdm(video_list):
    line = line.strip()
    video_info = line.split()
    
    video_file = video_info[0]  # MP4 文件名
    label_id = int(video_info[1])  # 动作的ID
    
    # 查找动作名称
    action_label = labels_dict.get(label_id, "unknown")  # 如果未找到匹配，返回 "unknown"
    
    # 获取视频的完整路径
    video_path = os.path.join(args.videos_path, video_file)
    video_name = os.path.splitext(video_file)[0]
    
    # 读取视频帧
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 将帧调整为合适的尺寸，比如 64x64
        frame_resized = cv2.resize(frame, (64, 64))
        frames.append(frame_resized)
    
    cap.release()
    
    # 将帧数组和标签保存为 .npz 文件
    if len(frames) > 0:
        frames = np.stack(frames)
        npz_filename = os.path.join(args.save_path, f"{video_name}.npz")
        np.savez_compressed(npz_filename, **{'frames': frames, 'label': action_label})

print("处理完成！")
