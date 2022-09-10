from data_utils.data_handling import load_data_training
from image_utils.video_handling import view_video

if __name__ == "__main__":
    video, keys = load_data_training("data/tmpiwj338u_")
    view_video(video, keys=keys, fps=60, is_bgr=False)
