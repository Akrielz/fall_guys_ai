from data_utils.data_handler import load_data_general
from image_utils.video_handler import view_video, load_video_iterator
from pipeline.data_loader import DataLoader

if __name__ == "__main__":
    file_name = "data/test/tmp5f0zshs6"
    keys = load_data_general(f"{file_name}.keys")

    # mask = None
    mask = DataLoader._balance_data_mask(keys)

    video = load_video_iterator(file_name=f"{file_name}.avi", mask=mask)

    view_video(video, keys=keys, fps=20, is_bgr=False, display_width=1920, display_height=1080)
