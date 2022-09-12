from data_utils.data_handler import load_data_training
from image_utils.video_handler import view_video

if __name__ == "__main__":
    video, keys = load_data_training("data/tmppnace70i")
    view_video(video, keys=keys, fps=15, is_bgr=False, font_size=2)
