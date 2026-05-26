import numpy as np

from .VideoDatasetReal import VideoDatasetReal


class VideoDatasetRealZeroPattern(VideoDatasetReal):
    def _loadpattern(self, video_path, pattern_name):
        return np.zeros((224, 224, 3), dtype=np.float32)
