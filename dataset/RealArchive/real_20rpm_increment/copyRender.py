import os
import glob
import shutil

# source folders
src_dir = "parameters"
src_vid_dir = "videos"

# destination folders
test_dst_dir = f"../real_20rpmincrement_5to5_test/{src_dir}"
test_dst_vid_dir = f"../real_20rpmincrement_5to5_test/{src_vid_dir}"

train_dst_dir = f"../real_20rpmincrement_5to5_train/{src_dir}"
train_dst_vid_dir = f"../real_20rpmincrement_5to5_train/{src_vid_dir}"

# make all dirs
os.makedirs(test_dst_dir, exist_ok=True)
os.makedirs(test_dst_vid_dir, exist_ok=True)
os.makedirs(train_dst_dir, exist_ok=True)
os.makedirs(train_dst_vid_dir, exist_ok=True)

# ---------------- JSON files ----------------
patterns = ["*renderA.json", "*renderB.json", "*renderC.json", "*renderD.json", "*renderE.json"]
selected_json = []
for pat in patterns:
    selected_json.extend(glob.glob(os.path.join(src_dir, pat)))

# copy test (renderA/renderF)
for f in selected_json:
    base = os.path.basename(f)
    target = os.path.join(test_dst_dir, base)
    shutil.copy2(f, target)
    # print(f"Copied {f} -> {target}")

# copy train (leftovers)
all_json = glob.glob(os.path.join(src_dir, "*.json"))
leftover_json = set(all_json) - set(selected_json)
for f in leftover_json:
    base = os.path.basename(f)
    target = os.path.join(train_dst_dir, base)
    shutil.copy2(f, target)
    # print(f"Copied leftover {f} -> {target}")

print("JSON Done.")

# ---------------- MP4 files ----------------
patterns = ["*renderA.mp4", "*renderB.mp4", "*renderC.mp4", "*renderD.mp4", "*renderE.mp4"]
selected_mp4 = []
for pat in patterns:
    selected_mp4.extend(glob.glob(os.path.join(src_vid_dir, pat)))

for f in selected_mp4:
    base = os.path.basename(f)
    target = os.path.join(test_dst_vid_dir, base)
    shutil.copy2(f, target)
    # print(f"Copied {f} -> {target}")

# copy train (leftovers)
all_mp4 = glob.glob(os.path.join(src_vid_dir, "*.mp4"))
leftover_mp4 = set(all_mp4) - set(selected_mp4)
print(len(leftover_mp4))
for f in leftover_mp4:
    base = os.path.basename(f)
    target = os.path.join(train_dst_vid_dir, base)
    shutil.copy2(f, target)
    # print(f"Copied leftover {f} -> {target}")

print("MP4 Done.")