import torch
import cv2
import json
import numpy as np
import os.path as osp
import albumentations as A
from torch.utils.data import Dataset

class VideoDatasetClassIntern(Dataset):
    def __init__(self, video_paths, para_paths, frame_num, time, aug_bool=False):
        self.video_paths = video_paths
        self.para_paths = para_paths
        self.frame_limit = int(frame_num * time)
        self.cluster_map = {i: i // 5 for i in range(50)}
        self.aug_bool = aug_bool

        self.augmentation = A.Compose([
            A.Perspective(scale=(0.02, 0.03), keep_size=True, p=0.6),
            A.MotionBlur(blur_limit=(3, 9), p=0.6),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        ])

        self.center_resize = A.Compose([
            A.Resize(224, 224, interpolation=1)
        ])

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __getitem__(self, index):
        name = self._loadname(self.para_paths[index])
        frames = self._loadvideo(self.video_paths[index])
        parameters, hotvector = self._loadparameters(self.para_paths[index])
        rpm = parameters[-1]
        return frames, parameters, hotvector, name, rpm

    def _loadvideo(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ids = list(range(1, total, 2))[-self.frame_limit:]  # take last odd frames

        frames = []
        for i in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if self.aug_bool:
            data = {f"image{i}": frames[i] for i in range(len(frames))}
            data["image"] = frames[0]
            out = self.augmentation(**data)
            frames_aug = [out[f"image{i}"] for i in range(len(frames))]
        else:
            frames_aug = [self.center_resize(image=f)["image"] for f in frames]

        frames_aug = [
            ((torch.tensor(f, dtype=torch.float32) / 255.0).permute(2, 0, 1) - self.mean) / self.std
            for f in frames_aug
        ]
        return torch.stack(frames_aug).permute(1, 0, 2, 3)  # (C, T, H, W)

    def _get_cluster(self, vis_idx):
        return torch.tensor(self.cluster_map[vis_idx])

    def _loadparameters(self, para_path):
        with open(para_path, 'r') as file:
            data = json.load(file)
        density = data["density"]
        surfT = data["surface_tension"]
        kinVisc = data["kinematic_viscosity"]
        rpm_index = data["rpm_idx"]
        cluster = self._get_cluster(data["visc_index"])
        return torch.tensor([density, surfT, kinVisc, rpm_index], dtype=torch.float32), cluster

    def _loadname(self, path):
        return osp.splitext(osp.basename(path))[0]

    def __len__(self):
        return len(self.video_paths)