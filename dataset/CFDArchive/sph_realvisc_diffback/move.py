# -*- coding: utf-8 -*-
import os
import json

# ----------------------------
# Settings
# ----------------------------
SRC_DIR = "parametersNorm"       # folder with original JSON files
DST_DIR = "../../fusion/newparamtersNorm"    # folder to save new JSON files
VISC_CLASS = 10                    # desired number of classes (e.g., 10)

# ----------------------------
# Build cluster map: 0..49 -> class 0..(VISC_CLASS-1)
# class_num = int(50 // visc_class); cluster_map[i] = i // class_num
# ----------------------------
if VISC_CLASS <= 0 or 50 % VISC_CLASS != 0:
    raise ValueError("VISC_CLASS must be a positive divisor of 50 (e.g., 10, 5, 2, 1).")
class_num = int(50 // VISC_CLASS)
cluster_map = {i: (i // class_num) for i in range(50)}  # e.g., 10 clusters when VISC_CLASS=10

# ----------------------------
# Ensure destination exists (Python 2 safe)
# ----------------------------
if not os.path.exists(DST_DIR):
    os.makedirs(DST_DIR)

# ----------------------------
# Process files
# ----------------------------
count_total = 0
count_written = 0

for fname in os.listdir(SRC_DIR):
    if not fname.endswith(".json"):
        continue
    src_path = os.path.join(SRC_DIR, fname)
    dst_path = os.path.join(DST_DIR, fname)

    # Load JSON
    try:
        with open(src_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print("Skip (read error): {} ({})".format(fname, e))
        continue

    count_total += 1

    # Get visc_index and map to class
    visc_index = data.get("visc_index", None)
    if visc_index is None:
        print("Skip (no visc_index): {}".format(fname))
        continue
    if not isinstance(visc_index, (int, long)) or visc_index < 0 or visc_index > 49:
        print("Skip (invalid visc_index={}): {}".format(visc_index, fname))
        continue

    # Apply mapping; add a new key so originals are untouched in place
    data["visc_class"] = int(cluster_map[visc_index])          # new 10-class label
    data["visc_class_binsize"] = int(class_num)                 # e.g., 5 when 10 classes
    data["visc_class_count"] = int(VISC_CLASS)                  # e.g., 10

    # Save new JSON to destination
    try:
        with open(dst_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        count_written += 1
        print("Wrote: {} -> class {}".format(fname, data["visc_class"]))
    except Exception as e:
        print("Write error: {} ({})".format(fname, e))

print("Done. Processed: {} files, wrote: {} files into '{}'.".format(count_total, count_written, DST_DIR))