# Mask mp4 video
import glob
import os
import os.path as osp
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from preprocess.mobile_sam import sam_model_registry, SamPredictor

class videoToMask():
    '''
    Masking mp4 train dataset videos using MobileSAM. 80ms/frame in masking,
    Input: (1024x1024) RGB videos
    Output: (1024x1024) greyscale masked videos
    '''
    def __init__(self, data_root, video_subdir, save_root, frame_num, checkpoint):
        self.data_root = data_root
        self.video_subdir = video_subdir
        self.save_root = save_root

        self.frame_num = frame_num
        self.checkpoint = checkpoint
        self.vortex_model = sam_model_registry["vit_t"](checkpoint=self.checkpoint)
        self.predictor = SamPredictor(self.vortex_model)

    def mask_videos(self):
        video_paths = glob.glob(osp.join(self.data_root, self.video_subdir, "*.mp4"))
        mask_paths = self.save_root
        self.vortex_model.eval()
        self.vortex_model.cuda()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 format
        video_num = 1
        
        for video_path in video_paths: # 4sec for video_writer,50ms/frame for masking
            frames, _, _, _ = self.__videoToImage__(video_path)
            video_writer = cv2.VideoWriter(osp.join(mask_paths, f"{video_num:05d}.mp4"), fourcc, self.frame_num, (256, 256), isColor=False)
            masks = self.__imageToMask__(frames)
            count=0
            for mask in masks:
                video_writer.write(mask)
                count+=1

            video_num += 1
            video_writer.release()
    
    def __videoToImage__(self, video_path):
        "capture frames and convert to image"
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) 
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # not used, but kept just in case
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # not used, but kept just in case
        
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            frame_count += 1

            if frame_count % (max(1, int(fps/self.frame_num))) == 0: # frame selection, FRAME_NUM per second.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.array(frame, dtype=np.float32)
                frames.append(frame)
        cap.release()
        frames = np.array(frames, dtype=np.float32) # Shape: [T, H, W, C]

        return frames, frame_width, frame_height, fps

    def __imageToMask__(self, frames):
        '''MobileSAM Masking applied'''
        masks =[]
        for frame in frames:
            self.predictor.set_image(image=frame, image_format="RGB") # normalize RGB, padding, make tensor
            mask, _, _ = self.predictor.predict(multimask_output = False, return_logits=True) # get (256, 256) mask with 0~255 uint8 format
            masks.append(mask) # Shape: [T, H, W]
        
        return masks
