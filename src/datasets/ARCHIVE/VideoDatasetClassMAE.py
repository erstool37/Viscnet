from torch.utils.data import Dataset
import cv2
import json
import numpy as np
import torch
import os.path as osp
import albumentations as A

class VideoDatasetClassMAE(Dataset):
    def __init__(self, video_paths, para_paths, frame_num, time, aug_bool=False):
        self.video_paths = video_paths
        self.para_paths = para_paths
        self.frame_limit = int(frame_num * time)
        self.cluster_map = {i: i // 5 for i in range(50)}  # 10 clusters
        self.aug_bool = aug_bool

        self.augmentation = A.Compose([
            A.Perspective(scale=(0.02, 0.03), keep_size=True, p=0.6),
            A.MotionBlur(blur_limit=(3, 9), p=0.6),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        ])

        self.center_resize = A.Compose([
            A.Resize(224, 224, interpolation=1)
        ])

    def __getitem__(self, index):
        names = self._loadname(self.para_paths[index])
        frames = self._loadvideo(self.video_paths[index])
        parameters, hotvector = self._loadparameters(self.para_paths[index])
        rpm = parameters[-1]
        print(frames.shape, parameters, hotvector, names, rpm)
        return frames, parameters, hotvector, names, rpm

    def _loadvideo(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 2 == 1:  # Keep only odd-numbered frames
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_idx += 1

        cap.release()
        frames = frames[-self.frame_limit:]

        if self.aug_bool:
            N = len(frames)
            data = {f"image{i}": frames[i] for i in range(N)}
            data["image"] = frames[0]  # dummy
            out = self.augmentation(**data)
            frames_aug = [out[f"image{i}"] for i in range(N)]
        else:
            frames_aug = [self.center_resize(image=f)["image"] for f in frames]

        frames_aug = [(f / 127.5 - 1.0).astype(np.float32) for f in frames_aug]  # normalize to [-1, 1]
        frames_tensor = torch.tensor(np.stack(frames_aug)).permute(3, 0, 1, 2)  # (C, T, H, W)
        return frames_tensor

    def _loadhotvector(self, cls):
        hot = torch.zeros(50, dtype=torch.float32)
        hot[cls] = 1.0
        return torch.tensor(hot)

    def _get_cluster(self, vis_idx: int) -> int:
        return torch.tensor(self.cluster_map[vis_idx])

    def _loadparameters(self, para_path):
        try:
            with open(para_path, 'r') as file:
                data = json.load(file)
                density = data["density"]
                surfT = data["surface_tension"]
                kinVisc = float(data["kinematic_viscosity"])
                rpm_index = int(data["rpm_idx"])
                hotvector = self._get_cluster(int(data["visc_index"]))
                return torch.tensor([density, surfT, kinVisc, rpm_index], dtype=torch.float32), hotvector
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON at: {para_path}")

    def _loadname(self, video_path):
        name = osp.splitext(osp.basename(video_path))
        return name[0]

    def __len__(self):
        print(len(self.video_paths))
        return len(self.video_paths)