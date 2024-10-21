import os
import json
import random
import torch
import torch.utils.data as data
from torch.utils.data import ConcatDataset
from PIL import Image
import numpy as np
from camera_utils import transform_pose, generate_rays_with_extrinsics, visualize_extrinsics
from glob import glob
from transformers import AutoTokenizer, AutoConfig
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms

class VLNCEDataset(data.Dataset):
    def __init__(self, configs, transform=None, return_pt=False, random=True, 
                 enable_time=True, enable_camera=True, enable_desc=True, **kwargs):
        self.configs = configs
        self.data_root = configs.data_path
        self.video_length = configs.num_frames
        self.img_size = configs.image_size
        self.mask_prob = configs.mask_prob
        self.transform = transform
        self.return_pt = return_pt
        self.mask_prefix = configs.mask_prefix
        self.random = random
        self.enable_time = enable_time
        self.enable_camera = enable_camera
        self.enable_desc = enable_desc

        self.data_all = self.load_data(self.data_root)

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        item = self.data_all[index]

        frames = self.load_images(item['image_paths'])
        camera_pose, rays = self.load_poses_and_rays(item['pose_paths'])
        prompt = item['prompt'] if self.enable_desc else ""
        
        if self.enable_time:
            mask = np.zeros(self.video_length, dtype=int)
            if self.random:
                mask[:np.random.randint(1, self.mask_prefix)] = 1
            else:
                mask[:self.mask_prefix] = 1
            mask = ~torch.tensor(mask, dtype=torch.bool)
        else:
            mask = torch.tensor(np.random.rand(self.video_length) < self.mask_prob, dtype=torch.bool)
            if torch.all(~mask):
                unmask_index = random.randint(0, self.video_length - 1)
                mask[unmask_index] = True
            if torch.all(mask):
                mask_index = random.randint(0, self.video_length - 1)
                mask[mask_index] = False

        return {
            'frames': frames,
            'camera_pose': camera_pose,
            'ray': rays,
            'mask': mask,
            'prompt': prompt,
            'enable_time': self.enable_time,
            'enable_camera': self.enable_camera,
        }

    def load_data(self, root_dir):
        data = []
        for id_folder in os.listdir(root_dir):
            id_path = os.path.join(root_dir, id_folder)
            if not os.path.isdir(id_path):
                continue
            
            result_json_path = os.path.join(id_path, 'result.json')
            sequence_infos_path = os.path.join(id_path, 'sequence_infos.json')

            if not os.path.exists(result_json_path) or not os.path.exists(sequence_infos_path):
                continue

            with open(result_json_path, 'r') as result_file, open(sequence_infos_path, 'r') as seq_info_file:
                results = json.load(result_file)
                seq_info = json.load(seq_info_file)

                for seq, desc in results.items():
                    image_paths = []
                    pose_paths = []
                    valid_segments = []

                    try:
                        start, end = map(int, seq.split('-'))
                        
                        for i in range(start, end + 1):
                            waypoint_path = os.path.join(id_path, f'waypoint_{i}')
                            if not os.path.exists(waypoint_path):
                                continue
                            img_path = sorted(glob(os.path.join(waypoint_path, "*.jpg")), 
                                              key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
                            pose_path = sorted(glob(os.path.join(waypoint_path, "*.txt")), 
                                               key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
                            image_paths.extend(img_path)
                            pose_paths.extend(pose_path)
                    except Exception as e:
                        print(f'Warning: parse error: {result_json_path}')

                    if len(image_paths) >= self.video_length:
                        for j in range(0, len(image_paths) - self.video_length + 1, self.video_length):
                            segment_image_paths = image_paths[j:j + self.video_length]
                            segment_pose_paths = pose_paths[j:j + self.video_length]
                            if len(segment_image_paths) == self.video_length:
                                valid_segments.append({
                                    'image_paths': segment_image_paths,
                                    'pose_paths': segment_pose_paths,
                                    'prompt': desc,
                                    'episode_id': seq_info['episode_id']
                                })
                        # 如果最后一段segment长度 < video_length，则重复最后一帧直到长度=video_length
                        remaining = len(image_paths) % self.video_length
                        if remaining != 0:
                            segment_image_paths = image_paths[-remaining:]
                            segment_pose_paths = pose_paths[-remaining:]
                            last_image = segment_image_paths[-1]
                            last_pose = segment_pose_paths[-1]
                            while len(segment_image_paths) < self.video_length:
                                segment_image_paths.append(last_image)
                                segment_pose_paths.append(last_pose)
                            valid_segments.append({
                                'image_paths': segment_image_paths,
                                'pose_paths': segment_pose_paths,
                                'prompt': desc,
                                'episode_id': seq_info['episode_id']
                            })
                        data.extend(valid_segments)
                    else:
                        # 复制image_paths的最后一个元素直到长度为video_length，将其插入valid_segments
                        if len(image_paths) > 0:
                            segment_image_paths = image_paths.copy()
                            segment_pose_paths = pose_paths.copy()
                            last_image = image_paths[-1]
                            last_pose = pose_paths[-1]
                            while len(segment_image_paths) < self.video_length:
                                segment_image_paths.append(last_image)
                                segment_pose_paths.append(last_pose)
                            valid_segments.append({
                                'image_paths': segment_image_paths,
                                'pose_paths': segment_pose_paths,
                                'prompt': desc,
                                'episode_id': seq_info['episode_id']
                            })
                            data.extend(valid_segments)
        return data

    def load_images(self, image_paths):
        images = []
        for path in image_paths:
            with Image.open(path) as img:
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        if self.return_pt:
            return torch.stack(images)
        else:
            return np.stack(images)

    def load_poses_and_rays(self, pose_paths):
        poses, rays = [], []
        # temp = []
        align_matrix = np.array([
                [-1, 0, 0, 0],  # x remains x
                [0, 0, 1, 0],  # z becomes y
                [0, -1, 0, 0],  # y becomes z
                [0, 0, 0, 1]   # homogeneous coordinates remain the same
            ]) + 1e-17
        # align_matrix = np.eye(4)
        reference_pose = None
        for pose_path in pose_paths:
            pose = np.loadtxt(pose_path).reshape(4, 4)
            pose = align_matrix @ pose
            if reference_pose is None:
                reference_pose = pose
            transformed_pose = transform_pose(reference_pose, pose)
            transformed_pose[:-1, 3] *= -1
            # transformed_pose[1, 3], transformed_pose[2, 3] \
            #     = transformed_pose[2, 3], transformed_pose[1, 3]
            pose = transformed_pose[:-1, 3].flatten()
            ray = generate_rays_with_extrinsics(transformed_pose, width=self.img_size, height=self.img_size)
            # temp.append(transformed_pose)
            poses.append(torch.tensor(pose, dtype=torch.float32))
            rays.append(torch.tensor(ray, dtype=torch.float32))
        # visualize_extrinsics(temp, './vis.jpg')
        if self.return_pt:
            return torch.stack(poses), torch.stack(rays)
        else:
            return poses, rays

class DataProcessor:
    def __init__(self, dataset):
        self.dataset = dataset

    def process_and_save(self, output_dir, sample_range):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for idx in tqdm(sample_range, desc="Processing: "):
            data_item = self.dataset[idx]
            # Get the prompt
            instruction = data_item['prompt']

            # Get the frames as (num_frame, H, W, C)
            frames = data_item['frames']  # Assuming frames are already in (num_frame, C, H, W) format
            frames = frames.astype(int)

            # Create the dictionary
            data_dict = {
                'instruction': instruction,
                'images': frames  # Convert tensor to numpy array
            }

            # Save as npz file
            output_path = os.path.join(output_dir, f"data_{idx}.npz")
            np.savez(output_path, **data_dict)

class Config:
    def __init__(self, data_path, image_size=64):
        # Dataset configuration
        self.data_path = data_path  # 数据集路径
        self.num_frames = 16  # 每个视频的帧数
        self.image_size = image_size  # 图片的大小 (H, W)
        self.mask_prob = 0.2  # 掩码概率
        self.mask_prefix = 3  # 前缀掩码长度
        self.random_mask = True  # 是否随机生成掩码
        self.return_pt = True  # 是否返回 PyTorch 张量
        self.enable_time = True  # 是否启用时间序列掩码
        self.enable_camera = True  # 是否启用相机位姿数据
        self.enable_desc = True  # 是否启用文本描述 (prompt)

        # Optional transformations (e.g., normalize or augment images)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ])

if __name__ == '__main__':
    r2r_config = Config(data_path="/mnt/HDD1_3TB/jinzhou/vlnce/processed_r2r")
    rxr_config = Config(data_path="/mnt/HDD1_3TB/jinzhou/vlnce/processed_rxr")
    dataset = ConcatDataset([
        VLNCEDataset(r2r_config, transform=r2r_config.transform),
        VLNCEDataset(rxr_config, transform=rxr_config.transform)
    ])
    # train_set, test_set = train_test_split(dataset, test_size=0.001, random_state=42)

    output_directory = "/mnt/HDD1_3TB/jinzhou/vlnce_processed/train"
    processor = DataProcessor(dataset)
    processor.process_and_save(output_directory, range(len(dataset)-100))

    output_directory = "/mnt/HDD1_3TB/jinzhou/vlnce_processed/test"
    processor.process_and_save(output_directory, range(len(dataset)-100, len(dataset)))
