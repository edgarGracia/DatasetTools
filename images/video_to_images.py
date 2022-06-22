from pathlib import Path
from tqdm import tqdm
import argparse
import math
import cv2



def video2img(video_path: Path, output_path: Path,
    skip_frames: int = 0, img_extension: str = '.jpg',
    start_frame: int = 0, end_frame: int = float("inf")):

    # Open video
    video = cv2.VideoCapture(str(video_path))
    ret, img = video.read()
    assert ret, f"Unable to read {str(video_path)}"
    
    # Calculate leading zeros
    length = min(
        end_frame,
        int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ) - start_frame

    if length > 0:
        leading_zeros = math.ceil(math.log10(length * 1.5))
    else:
        # Some codec return length 0, so we set a large number of zeros
        leading_zeros = 10
    
    args.output.mkdir(exist_ok=True, parents=True)

    f = 0
    if start_frame is not None:
        for i in range(start_frame):
            ret, img = video.read()
            f += 1
        assert ret, "Start frame is greater than video length"
    pbar = tqdm(total=length)
    while ret:
        img_path = output_path.joinpath(
            str(f).zfill(leading_zeros)+img_extension)

        cv2.imwrite(str(img_path), img)
        for _ in range(skip_frames+1):
            ret, img = video.read()
            f+=1
            if f >= end_frame:
                ret = False
        
        pbar.update(skip_frames+1)
        pbar.set_description(str(img_path))
    
    video.release()



if __name__ == "__main__":
    
    ap = argparse.ArgumentParser(
        description="Save the frames of a video in images")

    ap.add_argument("video", type=Path, help="Path to the video")
    ap.add_argument("output", type=Path, help="Output folder")
    ap.add_argument("-s", "--skip", type=int, default=0, help="Skip n frames")
    ap.add_argument("-e", "--extension", default=".jpg",
        help="Output image extension. e.g. '.jpg'")
    ap.add_argument("--start-frame", type=int, default=0, help="Start frame")
    ap.add_argument("--end-frame", type=int, default=float("inf"),
        help="End frame")
    
    args = ap.parse_args()
    
    video2img(args.video, args.output, args.skip, args.extension,
        args.start_frame, args.end_frame)