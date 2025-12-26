import os
import os.path as osp
import glob
import numpy as np
import cv2
from patchify import patchify
from preprocess.mobile_sam.utils import transforms

class PreprocessorSeg:
    '''GT mask erosion and dilation, patch extraction, and bounding box extraction for segmentation tasks.'''
    def __init__(self, data_root, image_subdir, mask_subdir, save_root, image_save_subdir, mask_save_subdir, box_save_subdir,
                 radius_erosion=1, iter_erosion=1, radius_dilation=3, iter_dilation=2):
        # Directories
        self.data_root = data_root
        self.image_subdir = image_subdir
        self.mask_subdir = mask_subdir
        self.save_root = save_root
        self.image_save_subdir = image_save_subdir
        self.mask_save_subdir = mask_save_subdir
        self.box_save_subdir = box_save_subdir

        # Parameters
        self.patch_size = 256
        self.patch_step = 256
        self.radius_erosion = radius_erosion
        self.iter_erosion = iter_erosion
        self.radius_dilation = radius_dilation
        self.iter_dilation = iter_dilation
        self.original_image_size = (256, 256)
        self.image_encoder_size = 1024
        self.resizer = transforms.ResizeLongestSide(self.image_encoder_size)

    def apply_morphology(self, mask, morph_op="opening"):
        """Applies erosion and dilation to the mask."""
        kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.radius_erosion, self.radius_erosion))
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.radius_dilation, self.radius_dilation))

        if morph_op == "opening":
            temp = cv2.erode(mask, kernel_erosion, iterations=self.iter_erosion)
            output = cv2.dilate(temp, kernel_dilation, iterations=self.iter_dilation)
        elif morph_op == "closing":
            temp = cv2.dilate(mask, kernel_dilation, iterations=self.iter_dilation)
            output = cv2.erode(temp, kernel_erosion, iterations=self.iter_erosion)
        else:
            raise ValueError("Invalid operation. Choose 'opening' or 'closing'.")
        return output

    def get_bounding_box(self, mask):
        """Extracts bounding box from a mask."""
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to box coordinates
        H, W = mask.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

        return np.array([x_min, y_min, x_max, y_max])

    def process_images(self):
        """Processes all images in the dataset."""
        image_paths = glob.glob(osp.join(self.data_root, self.image_subdir, "*.png"))
        
        for image_path in image_paths:
            mask_path = image_path.replace(self.image_subdir, self.mask_subdir)
            if not osp.exists(mask_path):
                print(f"Mask not found for {image_path}")
                continue

            image = cv2.imread(image_path)[:, :, :3]  # Ensure RGB
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            mask = self.apply_morphology(mask)
            img_patches = patchify(image, (self.patch_size, self.patch_size, 3), step=self.patch_step)
            mask_patches = patchify(mask, (self.patch_size, self.patch_size), step=self.patch_step)

            patch_cnt = 0
            for i in range(img_patches.shape[0]):
                for j in range(img_patches.shape[1]):
                    patch_img = self.resizer.apply_image(img_patches[i, j, 0, :, :, :])
                    patch_mask = mask_patches[i, j, :, :]
                    
                    if np.max(patch_mask) == 0: # Skip patches with no mask
                        continue
                    
                    box = self.resizer.apply_boxes(self.get_bounding_box(patch_mask), self.original_image_size)
                    
                    img_basename = int(osp.basename(image_path).replace(".png", "").replace("raw_", ""))
                    patch_img_save_path = osp.join(self.save_root, self.image_save_subdir, f"{img_basename:05d}_p{patch_cnt:01d}.png")
                    patch_mask_save_path = osp.join(self.save_root, self.mask_save_subdir, f"{img_basename:05d}_p{patch_cnt:01d}.png")
                    patch_box_save_path = osp.join(self.save_root, self.box_save_subdir, f"{img_basename:05d}_p{patch_cnt:01d}.npy")
                    
                    os.makedirs(osp.dirname(patch_img_save_path), exist_ok=True)
                    os.makedirs(osp.dirname(patch_mask_save_path), exist_ok=True)
                    os.makedirs(osp.dirname(patch_box_save_path), exist_ok=True)
                    
                    cv2.imwrite(patch_img_save_path, patch_img)
                    cv2.imwrite(patch_mask_save_path, patch_mask)
                    np.save(patch_box_save_path, box)
                    
                    patch_cnt += 1