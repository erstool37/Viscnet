import os
import cv2
import json
import time
import subprocess

# Updated viscosity and density
dynamic_viscosity = [0.89274, 1.5894, 2.8299, 5.0383, 8.9703, 15.971, 28.435, 50.626, 90.135, 160.48]
density = [996.89, 1048.4, 1090.3, 1124.1, 1151.4, 1173.8, 1192.3, 1207.9, 1221.1, 1232.5]

surface_tension = 0.0762
output_size = (512, 512)
max_frames = 50
target_fps = 10
base_dir = "mp4"
video_dir = os.path.join(base_dir, "videos")
param_dir = os.path.join(base_dir, "parameters")
temp_dir = "temp_frames"

# Setup directories
os.makedirs(video_dir, exist_ok=True)
os.makedirs(param_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

seen = set()
count = 0

while count < 100:
    mov_files = sorted([f for f in os.listdir('.') if f.endswith('.mov')])
    new_files = [f for f in mov_files if f not in seen]

    for f in new_files:
        if count >= 100:
            break
        try:
            cap = cv2.VideoCapture(f)
            if not cap.isOpened():
                print(f"Failed to open {f}")
                continue

            video_name = f"decay_5s_10fps_visc{count//10}_rpm{count%10}"
            frame_count = 0
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w, _ = frame.shape
                cropped = frame[104:h-104, 384:w-384]
                resized = cv2.resize(cropped, output_size)
                out_path = os.path.join(temp_dir, f"frame_{frame_count:03d}.png")
                cv2.imwrite(out_path, resized)
                frame_count += 1
            cap.release()

            mp4_path = os.path.join(video_dir, f"{video_name}.mp4")
            subprocess.run([
                'ffmpeg', '-y', '-framerate', str(target_fps),
                '-i', os.path.join(temp_dir, 'frame_%03d.png'),
                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', mp4_path
            ], check=True)

            metadata = {
                "height": 512,
                "width": 512,
                "fps": target_fps,
                "dynamic_viscosity": dynamic_viscosity[count//10],
                "density": density[count//10],
                "surface_tension": surface_tension
            }
            json_path = os.path.join(param_dir, f"{video_name}.json")
            with open(json_path, "w") as jf:
                json.dump(metadata, jf, indent=2)

            for ftemp in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, ftemp))

            print(f"[{count+1}/100] Done: {video_name}.mp4")
            seen.add(f)
            count += 1

        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("Waiting for new files...")
    time.sleep(30)