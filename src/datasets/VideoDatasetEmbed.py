from torch.utils.data import Dataset
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp

class VideoDatasetEmbed(Dataset):
    def __init__(self, video_paths, para_paths, frame_num, time):
        '''Initialize dataset'''
        self.video_paths = video_paths
        self.para_paths = para_paths
        self.frame_limit = frame_num * time

    def __getitem__(self, index):
        frames = self._loadvideo(self.video_paths[index], self.frame_limit)
        parameters = self._loadparameters(self.para_paths[index])
        names = self._loadname(self.para_paths[index])
        # rpm_idx = parameters[-1] 
        rpm = parameters[-1]
        return frames, parameters, names, rpm

    def _loadvideo(self, video_path, frame_limit):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.frame_limit:
            ret, frame = cap.read()
            if not ret:
                h, w, c = 512, 512, 3  # fallback shape
                pad = np.zeros((h, w, c), dtype=np.uint8)
                frames += [pad] * (self.frame_limit - len(frames))
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        frames = np.array(frames, dtype=np.float32) # required only for no masked
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
        
        # FOR RESNET34
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        return frames
    
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