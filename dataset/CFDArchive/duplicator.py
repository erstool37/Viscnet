import os
import glob
import shutil

# Directory containing your sim_####.json files
source_dir = 'parameters'

# # Find all files named sim_####.json
# for filepath in glob.glob(os.path.join(source_dir, 'sim_*.json')):
#     filename = os.path.basename(filepath)          
#     name, ext = os.path.splitext(filename)         # name="sim_1234", ext=".json"

#     # Create copies with prefixes A, B, C, D
#     # for prefix in ['A', 'B', 'C', 'D']:
#     #     new_filename = f"{prefix}{filename}"       # e.g. "Asim_1234.json"
#     #     new_path = os.path.join(source_dir, new_filename)
#     #     shutil.copy(filepath, new_path)

#     os.remove(filepath)

source_dir = 'sph_improved/videos'
dest_dir   = 'sph_selected/videos'

os.makedirs(dest_dir, exist_ok=True)

pattern = os.path.join(source_dir, '*50.mp4')
for src_path in glob.glob(pattern):
    shutil.copy(src_path, dest_dir)