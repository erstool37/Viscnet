from torch.utils.data import Dataset
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
import albumentations as A

class VideoDatasetReal(Dataset):
    def __init__(self, video_paths, para_paths, frame_num, time, aug_bool, visc_class):

        self.video_paths = video_paths
        self.para_paths = para_paths
        self.frame_limit = int(frame_num * time)
        self.class_num = 10 // visc_class
        self.cluster_map = {i: i // self.class_num for i in range(10)} # 10 clusters
        self.aug_bool = aug_bool

        self.augmentation = A.Compose([
            A.Perspective(scale=(0.01, 0.02), keep_size=True, p=0.6),
            A.MotionBlur(blur_limit=(3, 7), p=0.6),
            A.RandomBrightnessContrast(0.05, 0.1, p=0.5),
        ])

        self.center_resize = A.Compose([
            A.Resize(224, 224, interpolation=1)
        ])

    def __getitem__(self, index):
        names = self._loadname(self.para_paths[index])
        frames = self._loadvideo(self.video_paths[index])
        parameters, hotvector, pattern_name = self._loadparameters(self.para_paths[index])
        pattern = self._loadpattern(self.video_paths[index], pattern_name)
        rpm = parameters[-1]
        return frames, parameters, hotvector, names, rpm, pattern

    def _loadvideo(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames = frames[:self.frame_limit]

        if self.aug_bool:
            data = {f"image{i}": frames[i] for i in range(self.frame_limit)}
            data["image"] = frames[0]
            out = self.augmentation(**data)
            frames_aug = [out[f"image{i}"] for i in range(self.frame_limit)]
        else:
            frames_aug = [self.center_resize(image=f)["image"] for f in frames] # mostly for real images

        frames_aug = [(f / 127.5 - 1.0).astype(np.float32) for f in frames_aug]
        frames_tensor = torch.tensor(np.stack(frames_aug)).permute(0, 3, 1, 2)
        
        return frames_tensor
    
    def _get_cluster(self, vis_idx: int) -> int:
        return self.cluster_map[vis_idx]

    def _loadparameters(self, para_path):
        with open(para_path, 'r') as file:
            data = json.load(file)

            density = data["density"]
            surfT = data["surface_tension"]
            kinVisc = float(data["kinematic_viscosity"])
            rpm_index = int(data["rpm_idx"])
            hotvector = self._get_cluster(int(data["visc_index"]))
            pattern_name = str(data["background"])
            
        return torch.tensor([density, surfT, kinVisc, rpm_index], dtype=torch.float32), torch.tensor(hotvector), pattern_name    # kept this state for CE loss shape compatibility
    
    def _loadname(self, video_path):
        name = osp.splitext(osp.basename(video_path))
        return name[0]
    
    def _loadpattern(self, video_path, pattern_name):
        base_path = osp.dirname(osp.dirname(video_path))
        pattern_path = osp.join(base_path, "backgrounds", f"{pattern_name}.png")
        img = cv2.imread(pattern_path)
        if img is None:
            raise FileNotFoundError(pattern_path)
        pattern = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pattern = (pattern / 127.5 - 1.0).astype(np.float32)  # [-1, 1]
        return pattern

    def __len__(self):
        return len(self.video_paths)