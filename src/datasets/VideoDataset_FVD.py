from torch.utils.data import IterableDataset, Dataset
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp

class VideoDataset_FVD(Dataset):
    def __init__(self, video_paths, para_paths, frame_num, time):
        '''
        separate dataloader, for non-RESNET use, needed cuz we do not use the same pixel statistics
        also different permutation required
        '''
        self.video_paths = video_paths
        self.para_paths = para_paths
        self.frame_limit = frame_num * time

    def __getitem__(self, index):
        frames = self._loadvideo(self.video_paths[index], self.frame_limit)
        parameters = self._loadparameters(self.para_paths[index])
        names = self._loadname(self.para_paths[index])
        return frames, parameters, names

    def _loadvideo(self, video_path, frame_limit):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened() and len(frames) < self.frame_limit:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR → RGB
                frames.append(frame)
        cap.release()
        frames = np.array(frames, dtype=np.float32) / 255.0 # required only for no masked
        frames = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2)

        return frames
    
    def _loadparameters(self, para_path):
        with open(para_path, 'r') as file:
            data = json.load(file)
            density = data["density"]
            dynVisc = float(data["dynamic_viscosity"])
            surfT = float(data["surface_tension"])
            kinVisc = float(data["kinematic_viscosity"])
            
        return torch.tensor([density, dynVisc, surfT, kinVisc], dtype=torch.float32)
    
    def _loadname(self, video_path):
        name = osp.splitext(osp.basename(video_path))
            
        return name[0]

    def __len__(self):
        return len(self.video_paths)
    
    # for Iterable Dataset
    """
    def __iter__(self):
        for video_path, para_path in zip(self.video_paths, self.para_paths):
            frames = self.__loadvideo__(video_path, self.frame_limit)
            parameters = self.__loadparameters__(para_path)
            yield frames, parameters
    """