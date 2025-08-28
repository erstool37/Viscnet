import cv2
import os
from glob import glob

# Set your input and output directories
input_dir = 'videos'      # <-- Replace with your actual input path
output_dir = '32frame_videos'    # <-- Replace with your actual output path

os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

video_paths = glob(os.path.join(input_dir, '*.mp4'))

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, filename)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 2 == 1:
            out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

print("Done. Filtered videos saved to:", output_dir)