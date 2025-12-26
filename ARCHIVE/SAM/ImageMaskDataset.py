import glob
import os.path as osp
import numpy as np
from torch.utils.data import Dataset

ImageMaskPathItem = namedtuple("ImageMaskPathItem", ["image_path", "mask_path"])

class ImageMaskDataset(Dataset):
    def __init__(self, root, image_subdir, mask_subdir, transform=None):
        super().__init__()
        self.root = root
        self.image_subdir = image_subdir
        self.mask_subdir = mask_subdir
        self.transform = transform

        images = glob.glob(osp.join(root, image_subdir, "*.png"))
        images = sorted(images)
        masks = [image.replace(image_subdir, mask_subdir) for image in images]

        self.path_items = [ImageMaskPathItem(image_path=image, mask_path=mask) for image, mask in zip(images, masks)]
        self._sanity_check()
        print(f"Found {len(self.path_items)} image-mask pairs")

    def _sanity_check(self):
        for item in self.path_items:
            assert osp.exists(item.image_path), f"Image path {item.image_path} does not exist"
            assert osp.exists(item.mask_path), f"Mask path {item.mask_path} does not exist"

    def __getitem__(self, index):
        item = self.path_items[index]
        image = np.array(Image.open(item.image_path).convert("RGB"))
        mask = np.array(Image.open(item.mask_path).convert("L"))

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask

    def __len__(self):
        return len(self.path_items)