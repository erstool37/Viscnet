import bpy
import sys
import os

# get obj 
obj_path = sys.argv[-1]  # passed after "--"
obj_name = os.path.splitext(os.path.basename(obj_path))[0]
output_dir = "/path/to/output"
output_path = os.path.join(output_dir, f"{obj_name}.mp4")

# clear previous obj file
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith("obj_"): # assuming objects are named as "obj_00001"
        bpy.data.objects.remove(obj, do_unlink=True)

# import obj file
bpy.ops.import_scene.obj(filepath=obj_path)

obj = bpy.context.selected_objects[0]
obj.name = f"obj_{obj_name}" # Reassure the name
obj.location = (0, 0, 0)
obj.scale = (1, 1, 1)

# Inherits mostly from .blend file, but reassures video settings
scene = bpy.context.scene
scene.render.filepath = output_path
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'

scene.render.resolution_x = 248 # VideoMAE resolution
scene.render.resolution_y = 248 
scene.frame_start = 1
scene.frame_end = 60 # VideoMAE requires 16 frames
scene.render.fps = 30

scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

# Render
bpy.context.scene.frame_set(1) # set current frame to 1
bpy.ops.render.render(write_still=False, animation=True) # write all frames from start to end