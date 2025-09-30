import os
from pathlib import Path

# Mapping from RPM value → index
rpm_to_index = {
    0: 270,
    1: 290,
    2: 310,
    3: 330,
    4: 350,
    5: 370,
    6: 390,
    7: 410,
    8: 430,
    9: 450,
}

# Folder containing your files
base_dir = Path("videos")

# Process both mp4 and json files
for file in base_dir.glob("*.*"):
    if not (file.suffix.lower() in [".mp4", ".json"]):
        continue

    name = file.stem  # filename without extension
    for rpm_val, idx in rpm_to_index.items():
        target_str = f"rpm{rpm_val}"
        if target_str in name:
            new_name = name.replace(target_str, f"rpm{idx}")
            new_path = file.with_name(new_name + file.suffix)
            print(f"Renaming: {file.name} → {new_path.name}")
            os.rename(file, new_path)
            break