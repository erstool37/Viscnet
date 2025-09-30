import os
import glob
import shutil

# source folders
src_dir = "real_20rpm_increment_2500/parameters"
src_dir2 = "real_20rpm_increment_2500/parametersNorm"
src_vid_dir = "real_20rpm_increment_2500/videos"

# Train set folder setting
train_dst_dir = f"real_20rpm_increment_4back/{src_dir}"
train_dst_dir2 = f"real_20rpm_increment_4back/{src_dir2}"
train_dst_vid_dir = f"real_20rpm_increment_4back/{src_vid_dir}"

# Test set folder setting
test_dst_dir = f"real_20rpm_increment_1back/{src_dir}"
test_dst_dir2 = f"real_20rpm_increment_1back/{src_dir2}"
test_dst_vid_dir = f"real_20rpm_increment_1back/{src_vid_dir}"

# make all dir
os.makedirs(train_dst_dir, exist_ok=True)
os.makedirs(train_dst_dir2, exist_ok=True)
os.makedirs(train_dst_vid_dir, exist_ok=True)
os.makedirs(test_dst_dir, exist_ok=True)
os.makedirs(test_dst_dir2, exist_ok=True)
os.makedirs(test_dst_vid_dir, exist_ok=True)

# ---------------- JSON ㄴfiles ----------------
patterns = [
    "*renderA.json", "*renderB.json", "*renderC.json", "*renderD.json", "*renderE.json",
    "*renderF.json", "*renderG.json", "*renderH.json", "*renderI.json", "*renderJ.json",
    # "*renderK.json", "*renderL.json", "*renderM.json", "*renderN.json", "*renderO.json",
    "*renderP.json", "*renderQ.json", "*renderR.json", "*renderS.json", "*renderT.json",
    # "*renderU.json", "*renderV.json", "*renderW.json", "*renderX.json", "*renderY.json"
]

check_patterns = ["*renderK.json", "*renderL.json", "*renderM.json", "*renderN.json", "*renderO.json"]


selected_json = []
selected_jsonNorm = []
check_json = []
check_jsonNorm  = []

# Make json list for parameter/parameterNorm
for pat in patterns:
    selected_json.extend(glob.glob(os.path.join(src_dir, pat)))
    selected_jsonNorm.extend(glob.glob(os.path.join(src_dir2, pat)))

for pat in check_patterns:
    check_json.extend(glob.glob(os.path.join(src_dir, pat)))
    check_jsonNorm.extend(glob.glob(os.path.join(src_dir2, pat)))

print("Selected json num :", len(selected_json))
print("Selected jsonNorm num :", len(selected_jsonNorm))
print("Checkerboard json num :", len(check_json))
print("Checkerboard jsonNorm num :", len(check_jsonNorm))

# copy train parameter json
for f in selected_json:
    base = os.path.basename(f)
    target = os.path.join(train_dst_dir, base)
    shutil.copy2(f, target)
print("Finished train parameter dataset num :", len(selected_json))

# copy train parameterNorm json
for f in selected_jsonNorm:
    base = os.path.basename(f)
    target = os.path.join(train_dst_dir2, base)
    shutil.copy2(f, target)
print("Finished train parameterNorm dataset num :", len(selected_jsonNorm))

# copy test parameter json(leftovers)
all_json = glob.glob(os.path.join(src_dir, "*.json"))
leftover_json = set(all_json) - set(selected_json) - set(check_json)
for f in leftover_json:
    base = os.path.basename(f)
    target = os.path.join(test_dst_dir, base)
    shutil.copy2(f, target)
print("Finished test parameter dataset num :", len(leftover_json))

# copy test parameterNorm json(leftovers)
all_jsonNorm = glob.glob(os.path.join(src_dir2, "*.json"))
leftover_json_Norm = set(all_jsonNorm) - set(selected_jsonNorm) - set(check_jsonNorm)
for f in leftover_json_Norm:
    base = os.path.basename(f)
    target = os.path.join(test_dst_dir2, base)
    shutil.copy2(f, target)
print("Finished test parameterNorm dataset num :", len(leftover_json_Norm))

# ---------------- MP4 files ----------------
patternsVid = [
    "*renderA.mp4", "*renderB.mp4", "*renderC.mp4", "*renderD.mp4", "*renderE.mp4",
    "*renderF.mp4", "*renderG.mp4", "*renderH.mp4", "*renderI.mp4", "*renderJ.mp4",
    # "*renderK.mp4", "*renderL.mp4", "*renderM.mp4", "*renderN.mp4", "*renderO.mp4",
    "*renderP.mp4", "*renderQ.mp4", "*renderR.mp4", "*renderS.mp4", "*renderT.mp4",
    # "*renderU.mp4", "*renderV.mp4", "*renderW.mp4", "*renderX.mp4", "*renderY.mp4"
]

check_patternsVid = ["*renderK.mp4", "*renderL.mp4", "*renderM.mp4", "*renderN.mp4", "*renderO.mp4"]

selected_vid = []
check_vid = []

for pat in patternsVid:
    selected_vid.extend(glob.glob(os.path.join(src_vid_dir, pat)))

for pat in check_patternsVid:
    check_vid.extend(glob.glob(os.path.join(src_vid_dir, pat)))

# copy train mp4
for f in selected_vid:
    base = os.path.basename(f)
    target = os.path.join(train_dst_vid_dir, base)
    shutil.copy2(f, target)

print("Finished train video dataset num :", len(selected_vid))

# copy test mp4(leftovers)
all_vid = glob.glob(os.path.join(src_vid_dir, "*.mp4"))
leftover_vid = set(all_vid) - set(selected_vid) - set(check_vid)
for f in leftover_vid:
    base = os.path.basename(f)
    target = os.path.join(test_dst_vid_dir, base)
    shutil.copy2(f, target)
print("Finished test video dataset num :", len(leftover_vid))


# Destination roots
src_file = "statistics.json"
train_root = "real_20rpm_increment_4back"
test_root  = "real_20rpm_increment_1back"

# Ensure source exists
if not os.path.isfile(src_file):
    raise FileNotFoundError(f"{src_file} not found.")

# Copy to both train and test
shutil.copy2(src_file, train_root)
shutil.copy2(src_file, test_root)

print(f"Copied {src_file} to: {train_root} and {test_root}")