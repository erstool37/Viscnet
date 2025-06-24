"""
Render all the obj files in the OBJ_DIR using Blender. 
Adjust Batch Size checking for VRAM capacity.
"""

import os.path as osp
from glob import glob
import subprocess
from multiprocessing import Pool

OBJ_DIR = "obj"
BLEND = "main.blend"
SCRIPT = "render.py" 
BATCH_SIZE = 1 # Adjust for VRAM

obj_files = sorted(glob(osp.join(OBJ_DIR, "*.obj")))

# worker function, equivalent to the shell command
def render(obj_path):
    return subprocess.call(["blender-3.6.0-linux-x64/blender", "-b", BLEND, "--python", SCRIPT, "--", obj_path])

# Parallel execution, leveraging all VRAM using a process pool
if __name__ == "__main__":
    with Pool(processes=BATCH_SIZE) as pool: 
        pool.map(render, obj_files)