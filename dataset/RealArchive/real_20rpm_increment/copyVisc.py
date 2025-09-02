import os
import glob
import shutil

# source folders
src_dir = "parameters"
# src_dir = "parametersNorm"
src_vid_dir = "videos"

# destination folders
# test_dst_dir = f"../real_20rpm_increment_even_test/{src_dir}"
# test_dst_vid_dir = f"../real_20rpm_increment_even_test/{src_vid_dir}"
# train_dst_dir = f"../real_20rpm_increment_odd_train/{src_dir}"
# train_dst_vid_dir = f"../real_20rpm_increment_odd_train/{src_vid_dir}"

test_dst_dir = f"../real_20rpm_increment_extra_test/{src_dir}"
test_dst_vid_dir = f"../real_20rpm_increment_extra_test/{src_vid_dir}"
train_dst_dir = f"../real_20rpm_increment_extra_train/{src_dir}"
train_dst_vid_dir = f"../real_20rpm_increment_extra_train/{src_vid_dir}"

# make all dirs
for d in [test_dst_dir, test_dst_vid_dir, train_dst_dir, train_dst_vid_dir]:
    os.makedirs(d, exist_ok=True)

# viscosity substrings to look for
# visc_tags = [
#     "visc0.89273932",
#     "visc21.31028677",
#     "visc37.94111891",
#     "visc67.5508754",
#     "visc120.2684812",
# ]

visc_tags = [
    "000.89274",
    "015.97088",
    "021.31029",
    "028.43477",
    "037.94112",
    "050.62564",
    "067.55088",
    "090.13457",
]

# ---------------- JSON files ----------------
all_json = glob.glob(os.path.join(src_dir, "*.json"))
selected_json = [f for f in all_json if any(tag in f for tag in visc_tags)]
leftover_json = set(all_json) - set(selected_json)

# copy test json
for f in selected_json:
    shutil.copy2(f, os.path.join(test_dst_dir, os.path.basename(f)))
print(f"Copied {len(selected_json)} JSON files to test")

# copy train json
for f in leftover_json:
    shutil.copy2(f, os.path.join(train_dst_dir, os.path.basename(f)))
print(f"Copied {len(leftover_json)} JSON files to train")

# ---------------- MP4 files ----------------
all_mp4 = glob.glob(os.path.join(src_vid_dir, "*.mp4"))
selected_mp4 = [f for f in all_mp4 if any(tag in f for tag in visc_tags)]
leftover_mp4 = set(all_mp4) - set(selected_mp4)

# copy test mp4
for f in selected_mp4:
    shutil.copy2(f, os.path.join(test_dst_vid_dir, os.path.basename(f)))
print(f"Copied {len(selected_mp4)} MP4 files to test")

# copy train mp4
for f in leftover_mp4:
    shutil.copy2(f, os.path.join(train_dst_vid_dir, os.path.basename(f)))
print(f"Copied {len(leftover_mp4)} MP4 files to train")