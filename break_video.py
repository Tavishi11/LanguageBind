import sys
import types
import torchvision.transforms.functional as _F

# Patch removed torchvision module before languagebind import
_fake = types.ModuleType("torchvision.transforms.functional_tensor")
_fake.__dict__.update({k: getattr(_F, k) for k in dir(_F) if not k.startswith("_")})
sys.modules["torchvision.transforms.functional_tensor"] = _fake

from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from moviepy import VideoFileClip
import os
from pathlib import Path
from PIL import Image
import tqdm
import torch

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_BASE_PATH = '/home/datasets/ego4d_data/v2/full_scale/'
UID_FILE        = '/home/sax023/LanguageBind/ego4d_subset_ids.txt'
FPS_REQUIRED    = 8

prefix_vid = 'mini_video_'
prefix_img = 'frame_'


# ── Video processor ───────────────────────────────────────────────────────────
def process_video(video_path, fps_required):
    clip = VideoFileClip(video_path)
    fps_clip = clip.fps
    fps_required = min(fps_required, fps_clip)

    output_dir             = os.path.join(str(Path.home()), 'video_outputs', Path(video_path).name.split('.')[0], f"fps_{fps_required}")
    output_dir_frames      = os.path.join(output_dir, 'frames')
    output_dir_mini_videos = os.path.join(output_dir, 'mini_videos')

    mini_video_files = []
    images = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir_frames)
        os.makedirs(output_dir_mini_videos)

    # Check receipt — skip if already processed
    try:
        with open(f'{output_dir}/vid_details.txt', 'r') as file:
            file_contents = file.read()
            files = file_contents.split('\n')
            for f in files:
                parts = f.split(',')
                if len(parts) < 2:
                    continue
                if 'video' in parts[1]:
                    vid_max_num = int(parts[1].strip().split('.')[0][len(prefix_vid):])
                    mini_video_files = [os.path.join(output_dir_mini_videos, f"mini_video_{i}.mp4") for i in range(vid_max_num + 1)]
                if 'frame' in parts[1]:
                    img_max_num = int(parts[1].strip().split('.')[0][len(prefix_img):])
                    images = [os.path.join(output_dir_frames, f"frame_{i}.jpg") for i in range(img_max_num + 1)]
        print(f"  ✓ Already done — skipping")
        clip.close()
        return mini_video_files, images, output_dir
    except:
        pass  # receipt not found — process below

    print(f"  Processing video segments...")
    num_secs = 8 // fps_required

    if num_secs == 0:  # fps_required > 8
        num_secs = 8 / fps_required
        start_time = 0
        end_time = num_secs
        keep_adding = True
        i = 0
        while keep_adding:
            mini_video = clip.subclipped(start_time, end_time).with_fps(fps_required)
            file_name = f"{output_dir_mini_videos}/mini_video_{i}.mp4"
            mini_video.write_videofile(file_name, fps=fps_required, logger=None)
            mini_video_files.append(file_name)
            start_time = end_time
            if start_time >= clip.duration:
                keep_adding = False
            else:
                end_time = min(start_time + num_secs, clip.duration)
            i += 1
    else:
        for i in range(0, int(clip.duration), num_secs):
            start_time = i
            end_time = min(i + num_secs, clip.duration)
            mini_video = clip.subclipped(start_time, end_time)
            if end_time == clip.duration:
                mini_video = mini_video.with_duration(end_time - start_time).with_fps(fps_required)
            else:
                mini_video = mini_video.with_duration(num_secs).with_fps(fps_required)
            file_name = f"{output_dir_mini_videos}/mini_video_{i//num_secs}.mp4"
            mini_video.write_videofile(file_name, fps=fps_required, logger=None)
            mini_video_files.append(file_name)

    print(f"  Extracting frames...")
    frames = clip.iter_frames(fps=fps_required)
    for i, f in tqdm.tqdm(enumerate(frames), desc="  Frames"):
        file_name = f"{output_dir_frames}/frame_{i}.jpg"
        Image.fromarray(f).save(file_name)
        images.append(file_name)

    print(f"  {len(mini_video_files)} segments, {len(images)} frames")

    with open(f'{output_dir}/vid_details.txt', 'w') as file:
        file.write(f'{output_dir_mini_videos}, mini_video_{len(mini_video_files)-1}.mp4\n'
                   f'{output_dir_frames}, frame_{len(images)-1}.jpg')

    clip.close()
    return mini_video_files, images, output_dir


# ── Main — process all 439 videos ────────────────────────────────────────────
def main():
    with open(UID_FILE) as f:
        uids = [line.strip().replace('.mp4', '') for line in f if line.strip()]

    print(f"Total videos: {len(uids)}")
    already_done = len(os.listdir(os.path.join(str(Path.home()), 'video_outputs'))) \
                   if os.path.exists(os.path.join(str(Path.home()), 'video_outputs')) else 0
    print(f"Already processed: {already_done}\n")

    for idx, uid in enumerate(uids):
        video_path = os.path.join(VIDEO_BASE_PATH, f"{uid}.mp4")
        print(f"[{idx+1}/{len(uids)}] {uid}")

        if not os.path.exists(video_path):
            print(f"  ✗ File not found — skipping")
            continue

        try:
            process_video(video_path, FPS_REQUIRED)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    total_done = len(os.listdir(os.path.join(str(Path.home()), 'video_outputs')))
    print(f"\nDone. {total_done}/{len(uids)} videos processed.")


if __name__ == '__main__':
    main()
