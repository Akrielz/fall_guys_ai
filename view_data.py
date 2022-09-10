from data_utils.data_handler import load_data_training
from image_utils.video_handler import view_video

if __name__ == "__main__":
    video, keys = load_data_training("data/tmpcxbvc8im")
    view_video(video, keys=keys, fps=20, is_bgr=False)
