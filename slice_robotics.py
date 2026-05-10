# import os
# from moviepy.video.io.VideoFileClip import VideoFileClip
# from pathlib import Path

# # Your labeled data
# eval_tasks = [
#     # P17 Tasks
#     {"run": "P17_PrizeRun", "file": "camera1-1024x768-004.mp4", "start": "07:54", "end": "07:57", "label": "wheelbarrow"},
#     {"run": "P17_PrizeRun", "file": "camera1-1024x768-005.mp4", "start": "02:08", "end": "02:14", "label": "wheelbarrow"},
#     {"run": "P17_PrizeRun", "file": "camera1-1024x768-005.mp4", "start": "02:27", "end": "02:36", "label": "blue barrel"},
#     {"run": "P17_PrizeRun", "file": "camera0-1024x768-004.mp4", "start": "04:10", "end": "04:25", "label": "stairs"},
    
#     # P16 Tasks
#     {"run": "P16_PrelimRun2", "file": "camera0-1024x768-004.mp4", "start": "01:39", "end": "01:43", "label": "black barrels"},
#     {"run": "P16_PrelimRun2", "file": "camera0-1024x768-003.mp4", "start": "01:29", "end": "01:34", "label": "door knockdown"},
#     {"run": "P16_PrelimRun2", "file": "camera0-1024x768-003.mp4", "start": "01:43", "end": "01:48", "label": "bamboo tree"},
#     {"run": "P16_PrelimRun2", "file": "camera0-1024x768-001.mp4", "start": "07:25", "end": "07:32", "label": "spot enter tunnel"},
#     {"run": "P16_PrelimRun2", "file": "camera0-1024x768-002.mp4", "start": "00:10", "end": "00:17", "label": "ATR enter tunnel"},
    
#     # P14 Tasks
#     {"run": "P14_PrizeRun", "file": "camera0-1024x768-002.mp4", "start": "04:30", "end": "04:33", "label": "drill"},
#     {"run": "P14_PrizeRun", "file": "camera0-1024x768-002.mp4", "start": "06:38", "end": "06:40", "label": "drill"},
#     {"run": "P14_PrizeRun", "file": "camera0-1024x768-003.mp4", "start": "07:06", "end": "07:09", "label": "fire extinguisher"},
#     {"run": "P14_PrizeRun", "file": "camera0-1024x768-003.mp4", "start": "09:26", "end": "09:29", "label": "fire extinguisher"},
#     {"run": "P14_PrizeRun", "file": "camera0-1024x768-004.mp4", "start": "00:00", "end": "00:03", "label": "path blocked by rocks"},
#     {"run": "P14_PrizeRun", "file": "camera0-1024x768-002.mp4", "start": "03:57", "end": "04:02", "label": "three-way junction"}
# ]

# VIDEO_ROOT = os.path.expanduser("~/research_data/videos")
# EXTRACT_DIR = os.path.expanduser("~/research_data/eval_extracts")
# os.makedirs(EXTRACT_DIR, exist_ok=True)

# def time_to_sec(t_str):
#     m, s = map(int, t_str.split(':'))
#     return m * 60 + s

# print(f"Starting Video Slicing Phase...")

# for task in eval_tasks:
#     input_path = os.path.join(VIDEO_ROOT, task['run'], task['file'])
    
#     # Create a unique, descriptive filename
#     # Example: P17_PrizeRun_wheelbarrow_07-54.mp4
#     clean_start = task['start'].replace(':', '-')
#     output_filename = f"{task['run']}_{task['label']}_{clean_start}.mp4"
#     output_path = os.path.join(EXTRACT_DIR, output_filename)

#     if os.path.exists(output_path):
#         print(f"Skipping: {output_filename} (already exists)")
#         continue

#     if os.path.exists(input_path):
#         print(f"Slicing: {output_filename}...")
#         try:
#             with VideoFileClip(input_path) as video:
#                 start_s = time_to_sec(task['start'])
#                 end_s = time_to_sec(task['end'])
#                 # MoviePy 2.0+ syntax
#                 sub = video.subclipped(start_s, end_s)
#                 sub.write_videofile(output_path, codec="libx264", audio=False, logger=None)
#         except Exception as e:
#             print(f"Error processing {output_filename}: {e}")
#     else:
#         print(f"Source Missing: {input_path}")

# print(f"\nDone! All extracts are in {EXTRACT_DIR}")

import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy import VideoFileClip, concatenate_videoclips

EXTRACT_DIR = os.path.expanduser("~/research_data/eval_extracts")

# Define the merge groups
# The key will be the NEW filename, the list contains the OLD filenames to be deleted
clip_groups = {
    "P17_PrizeRun_wheelbarrow_merged.mp4": [
        "P17_PrizeRun_wheelbarrow_02-08.mp4",
        "P17_PrizeRun_wheelbarrow_07-54.mp4"
    ],
    "P16_PrelimRun2_robots_entering_tunnel_merged.mp4": [
        "P16_PrelimRun2_spot enter tunnel_07-25.mp4",
        "P16_PrelimRun2_ATR enter tunnel_00-10.mp4"
    ],
    "P14_PrizeRun_fire_extinguisher_merged.mp4": [
        "P14_PrizeRun_fire extinguisher_07-06.mp4",
        "P14_PrizeRun_fire extinguisher_09-26.mp4"
    ],
    "P14_PrizeRun_drill_merged.mp4": [
        "P14_PrizeRun_drill_04-30.mp4",
        "P14_PrizeRun_drill_06-38.mp4"
    ]
}

def main():
    print(f"--- Starting Merge and Cleanup in {EXTRACT_DIR} ---")
    
    for merged_filename, source_files in clip_groups.items():
        print(f"\nProcessing group for: {merged_filename}")
        
        clips = []
        valid_sources = []
        
        for f in source_files:
            path = os.path.join(EXTRACT_DIR, f)
            if os.path.exists(path):
                print(f"  Found source: {f}")
                clips.append(VideoFileClip(path))
                valid_sources.append(path)
            else:
                print(f"  Missing source: {f}")

        if clips:
            # 1. Concatenate the clips
            print(f"  Merging {len(clips)} clips...")
            final_clip = concatenate_videoclips(clips, method="compose")
            output_path = os.path.join(EXTRACT_DIR, merged_filename)
            
            # 2. Write the merged file
            final_clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
            
            # 3. Close objects to release file locks
            for c in clips:
                c.close()
            final_clip.close()
            
            # 4. Remove the original fragments
            print(f"  Merge successful. Deleting source fragments...")
            for path in valid_sources:
                try:
                    os.remove(path)
                    print(f"    Deleted: {os.path.basename(path)}")
                except Exception as e:
                    print(f"    Error deleting {path}: {e}")
        else:
            print(f"  Skipping group: No valid sources found.")

    print("\n--- Process Complete ---")

if __name__ == "__main__":
    main()