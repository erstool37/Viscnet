from torch.utils.data import IterableDataset, Dataset
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
# from transformers import VideoMAEImageProcessor
from transformers import VivitImageProcessor
import albumentations as A

class VideoDatasetTrans(Dataset):
    def __init__(self, video_paths, para_paths, frame_num, time, aug_bool=False):
        '''Initialize dataset'''
        self.video_paths = video_paths
        self.para_paths = para_paths
        self.frame_limit = frame_num * time
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
        parameters = self._loadparameters(self.para_paths[index])
        names = self._loadname(self.para_paths[index])
        # rpm_idx = parameters[-1] 
        rpm = parameters[-1]
        return frames, parameters, names, rpm

    def _loadvideo(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0
        selected_count = 0

        while selected_count < 16:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= 32:
                break
            if frame_idx % 2 == 0:  # even-numbered frame
                small = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                if self.aug_bool:
                    small = self.augmentation(image=small)["image"]
                frames.append(small)
                selected_count += 1
            frame_idx += 1
        cap.release()
        
        preprocessed = self.processor(images=frames, return_tensors="pt") # only implemented for 50frame set videos, not ready for inference(versatile fps videos)
        print(preprocessed["pixel_values"].shape)
        return preprocessed["pixel_values"].squeeze(0)
    
    def _loadparameters(self, para_path):
        try :
            with open(para_path, 'r') as file:
                data = json.load(file)
                density = data["density"]
                dynVisc = float(data["dynamic_viscosity"])
                surfT = float(data["surface_tension"])
                kinVisc = float(data["kinematic_viscosity"])
                # rpm_index = int(data["rpm_idx"])
                rpm = int(data["rpm"])
            return torch.tensor([density, dynVisc, surfT, kinVisc, rpm], dtype=torch.float32)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON at: {para_path}")
    
    def _loadname(self, video_path):
        name = osp.splitext(osp.basename(video_path))
        return name[0]

    def __len__(self):
        return len(self.video_paths)