import os, glob, re

# root folders
VIDEO_DIR = "real_20rpm_increment_odd_train/videos"
PARA_DIR = "real_20rpm_increment_odd_train/parameters"
NORM_DIR = "real_20rpm_increment_odd_train/parametersNorm"

def rename_files_with_visc(folder, ext):
    files = glob.glob(os.path.join(folder, f"*.{ext}"))
    for f in files:
        base = os.path.basename(f)

        # look for viscosity pattern
        match = re.search(r"visc([0-9]*\.?[0-9]+)", base)
        if not match:
            continue

        visc_val = float(match.group(1))
        visc_str = f"{visc_val:09.5f}"   # ensures 000.00000 format

        # replace old visc number with formatted one
        new_base = re.sub(r"visc[0-9]*\.?[0-9]+", f"visc{visc_str}", base)
        new_path = os.path.join(folder, new_base)

        # rename file
        os.rename(f, new_path)
        print(f"Renamed {base} -> {new_base}")

# Apply to both JSON and MP4
rename_files_with_visc(VIDEO_DIR, "mp4")
rename_files_with_visc(PARA_DIR, "json")
rename_files_with_visc(NORM_DIR, "json")