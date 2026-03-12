# for the forensicface++

# import argparse
# import cv2
# import os
# from pathlib import Path

# base_dir = Path(__file__).resolve().parent.parent

# parser = argparse.ArgumentParser(description="Extract frames from videos.")
# parser.add_argument(
#     "--label",
#     choices=["real", "fake"],
#     default="real",
#     help="Video subfolder to read from and dataset subfolder to write to.",
# )
# args = parser.parse_args()

# video_folder = base_dir / "videos" / args.label
# output_folder = base_dir / "dataset" / args.label

# if not video_folder.exists():
#     raise FileNotFoundError(f"Video folder not found: {video_folder}")

# # makedirs = create a new directory if not there
# os.makedirs(output_folder, exist_ok=True)

# # listdir = check inside the file 
# for video in os.listdir(video_folder):

#     # path.join = is like the "smart glue" used to combine folder names and file names into a single path.
#     video_path = video_folder / video

#     # VideoCapture = core function of the opencv to read the video data
#     capture = cv2.VideoCapture(video_path)

#     frame_count = 0
#     while True: 
#         ret, frame = capture.read()
#         if not ret:
#             break

#         if frame_count % 30 == 0:
#             filename = f"{video}_{frame_count}.jpg"
#             cv2.imwrite(str(output_folder / filename), frame)
#         frame_count += 1
#     capture.release()

# print("Frames extracted successfully!")


# for the celeb

import argparse
import cv2
import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description="Extract frames from videos.")
parser.add_argument(
    "--label",
    choices=["real", "fake"],
    default="real",
    help="Video subfolder to read from and dataset subfolder to write to.",
)
parser.add_argument(
    "--subfolders",
    nargs="*",
    help="Optional list of subfolders under videos/<label> to process. If omitted, all subfolders are used.",
)
args = parser.parse_args()

video_folder = base_dir / "videos" / args.label
output_folder = base_dir / "dataset" / args.label

if not video_folder.exists():
    raise FileNotFoundError(f"Video folder not found: {video_folder}")

# makedirs = create a new directory if not there
os.makedirs(output_folder, exist_ok=True)

# listdir = check inside the file 
subfolders = args.subfolders if args.subfolders else os.listdir(video_folder)

for subfolder in subfolders:
    subfolder_path = video_folder / subfolder
    if not subfolder_path.exists():
        raise FileNotFoundError(f"Subfolder not found: {subfolder_path}")

    for video in os.listdir(subfolder_path):
        video_path = subfolder_path / video

        capture = cv2.VideoCapture(str(video_path))
        frame_count = 0

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            if frame_count % 30 == 0:
                filename = f"{subfolder}_{video}_{frame_count}.jpg"
                cv2.imwrite(str(output_folder / filename), frame)

            frame_count += 1

        capture.release()
