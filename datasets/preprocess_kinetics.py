import os
import numpy as np
import cv2  # OpenCV 用于视频处理
import pandas as pd
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--videos_path', type=str, required=True)  # 训练视频文件的根路径
parser.add_argument('--train_list', type=str, required=True)   # kinetics700_train_list_videos.txt 文件路径
parser.add_argument('--test_list', type=str, required=True)    # kinetics700_val_list_videos.txt 文件路径
parser.add_argument('--labels_file', type=str, required=True)  # kinetics_700_labels.csv 文件路径
parser.add_argument('--save_path', type=str, required=True)    # 处理后文件保存路径
args = parser.parse_args()

# 创建保存训练集和测试集的文件夹
train_save_path = os.path.join(args.save_path, "train")
test_save_path = os.path.join(args.save_path, "test")
os.makedirs(train_save_path, exist_ok=True)
os.makedirs(test_save_path, exist_ok=True)

# 读取标签文件并创建 ID 到标签名的映射
labels_df = pd.read_csv(args.labels_file)
labels_dict = labels_df.set_index('id').to_dict()['name']  # 创建 id -> 动作标签的映射

# 定义处理视频并保存为 .npz 文件的函数
def process_videos(video_list, video_root_path, save_path, labels_dict):
    counter = 0
    for line in tqdm(video_list):  # 逐行处理视频
        line = line.strip()
        video_info = line.split()
        
        video_file = video_info[0]  # 获取 MP4 文件名
        label_id = int(video_info[1])  # 获取动作 ID
        
        # 根据 ID 查找对应的动作标签
        action_label = labels_dict.get(label_id, "unknown")  # 如果未找到标签，默认返回 "unknown"
        
        # 获取视频的完整路径
        video_path = os.path.join(video_root_path, video_file)
        
        # 读取视频帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame.dtype != np.uint8:
                frame = (255 * (frame - frame.min()) / (frame.max() - frame.min())).astype(np.uint8)
                print("ATTENTION!!!")
            # 将帧调整为 64x64 像素
            frame_resized = cv2.resize(frame, (64, 64))
            frames.append(frame_resized)  # 保持 (H, W, C) 的顺序
        
        cap.release()
        
        # 如果视频帧非空，将其保存为 .npz 文件
        if len(frames) > 0:
            frames = np.stack(frames)  # 将帧堆叠为形状 (frames, H, W, C)
            traj_filename = f"traj_{counter:05d}"
            npz_filename = os.path.join(save_path, f"{traj_filename}.npz")
            # 保存帧和对应的动作标签
            np.savez_compressed(npz_filename, images=frames, instruction=action_label)
            counter += 1

# 处理训练集
print("正在处理训练集视频...")
with open(args.train_list, 'r') as f:
    train_video_list = f.readlines()
process_videos(train_video_list, args.videos_path, train_save_path, labels_dict)

# 处理测试集
print("正在处理测试集视频...")
with open(args.test_list, 'r') as f:
    test_video_list = f.readlines()
process_videos(test_video_list, args.videos_path, test_save_path, labels_dict)

print("处理完成！")
