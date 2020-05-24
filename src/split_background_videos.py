import cv2
import os
import argparse
from datetime import datetime, timedelta

SUPPORTED_VIDEO_EXTS = (".avi", ".mp4")
FFMPEG_COMMAND_STARTER = "ffmpeg -i {} "
FFMPEG_COMMAND_ADD_LINE = " -ss {} -t {} {} "
OUTPUT_FILE_FORMAT = "{}.avi"
DATETIME_FORMAT = "%H:%M:%S"

file_idx = None


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-videos-dir", type=str,
                        default="./data/input/bg0523")
    parser.add_argument("--output-videos-dir", type=str,
                        default="./data/input/bgclips0523")
    parser.add_argument("--seconds-per-sample", type=int, default=3)

    return parser.parse_args()


def _handle_single_video(file_name, args):
    if not file_name.endswith(SUPPORTED_VIDEO_EXTS):
        print("Unsupported video file {}".format(file_name))
        return

    if not os.path.exists(args.output_videos_dir):
        os.makedirs(args.output_videos_dir)

    global file_idx
    video_path = os.path.join(args.input_videos_dir, file_name)
    cap = cv2.VideoCapture(video_path)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) /
                   cap.get(cv2.CAP_PROP_FPS))
    s = 0
    delta = timedelta(seconds=args.seconds_per_sample)
    start_time = datetime(1991, month=1, day=1, hour=0, minute=0, second=0)
    cmd = FFMPEG_COMMAND_STARTER.format(video_path)

    while s + args.seconds_per_sample < duration:
        cmd += FFMPEG_COMMAND_ADD_LINE.format(
            start_time.strftime(DATETIME_FORMAT),
            args.seconds_per_sample,
            os.path.join(args.output_videos_dir,
                         OUTPUT_FILE_FORMAT.format(file_idx)),
        )
        file_idx += 1
        start_time += delta
        s += args.seconds_per_sample

    print(cmd)
    os.system(cmd)


def _get_start_id(cur_dir):
    max_id = 0
    for file_name in os.listdir(cur_dir):
        try:
            idx = int(file_name)
            max_id = max(idx, max_id)
        except:
            pass
    return max_id + 1


def main(args):
    if not os.path.exists(args.output_videos_dir):
        os.makedirs(args.output_videos_dir)
    if not os.path.isdir(args.input_videos_dir) \
            or not os.path.isdir(args.output_videos_dir):
        raise ValueError("unknown input/output videos dir {}/{}".format(
            args.input_videos_dir, args.output_videos_dir))
    global file_idx
    file_idx = _get_start_id(args.output_videos_dir)
    for file_name in os.listdir(args.input_videos_dir):
        _handle_single_video(file_name, args)


if __name__ == '__main__':
    main(_parse_args())
