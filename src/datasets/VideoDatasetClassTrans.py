from torch.utils.data import IterableDataset, Dataset
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
from transformers import VivitImageProcessor
import albumentations as A
from collections import deque

class VideoDatasetClassTrans(Dataset):
    def __init__(self, video_paths, para_paths, frame_num, time, aug_bool=False):
        '''Initialize dataset'''
        self.video_paths = video_paths
        self.para_paths = para_paths
        self.frame_limit = int(frame_num * time)
        self.cluster_map = {i: i // 5 for i in range(50)}
        self.processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        # self.processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)

        self.aug_bool = aug_bool
        self.augmentation = A.Compose([
            A.GaussNoise(var_limit=(10,50), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
        ])

    def __getitem__(self, index):
        frames = self._loadvideo(self.video_paths[index])
        parameters, hotvector = self._loadparameters(self.para_paths[index])
        names = self._loadname(self.para_paths[index])
        # rpm_idx = parameters[-1] 
        rpm = parameters[-1]
        return frames, parameters, hotvector, names, rpm

    def _loadvideo(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.aug_bool:
                frame = self.augmentation(image=frame)["image"]
            # print(frame.shape)
            # frame = self.processor(images=frame, do_normalize = True, return_tensors="np", input_data_format = "channels_last")["pixel_values"].squeeze(0).squeeze(0)
            frame = frame / 127.5 - 1.0
            frames.append(frame)
        cap.release()

        frames = np.array(frames[-self.frame_limit:], dtype=np.float32)
        frames = torch.tensor(frames).permute(0, 3, 1, 2)
        
        return frames

    def _loadhotvector(self, cls):
        hot = torch.zeros(50, dtype=torch.float32)
        hot[cls] = 1.0
        return torch.tensor(hot)

    def _get_cluster(self, vis_idx: int) -> int:
        return self.cluster_map[vis_idx]

    def _loadparameters(self, para_path):
        try :
            with open(para_path, 'r') as file:
                data = json.load(file)
                density = data["density"]
                # dynVisc = float(data["dynamic_viscosity"])
                # hotvector = self._loadhotvector(data["visc_index"])
                hotvector = self._get_cluster(int(data["visc_index"]))
                # hotvector = data["visc_index"]
                surfT = (data["surface_tension"])
                kinVisc = float(data["kinematic_viscosity"])
                # rpm_index = int(data["rpm_idx"])
                rpm = int(data["rpm"])
            return torch.tensor([density, surfT, kinVisc, rpm], dtype=torch.float32), hotvector
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON at: {para_path}")
    
    def _loadname(self, video_path):
        name = osp.splitext(osp.basename(video_path))
        return name[0]

    def __len__(self):
        print(len(self.video_paths))
        return len(self.video_paths)