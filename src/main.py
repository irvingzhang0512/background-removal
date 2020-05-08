import cv2
import os
import argparse

from datetime import datetime
from tqdm import tqdm
from deeplab_v3 import DeepLabV3

DEFAULT_BACKGROUND_COLOR = (255, 255, 255)
OUTPUT_IMG_FORMAT = ".png"
OUTPUT_VIDEO_FORMAT = ".avi"
OUTPUT_CV2_FOURCC = cv2.VideoWriter_fourcc(* 'XVID')
OUTPUT_FILE_NAME_FORMAT = "{}_{}_{}{}"  # origin_name/background/time/ext


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="deeplab_v3")
    parser.add_argument("--ckpt-path", type=str,
                        default="/ssd4/zhangyiyang/background-removal/data/deeplab_v3/xception_model")
    parser.add_argument("--gpu-devices", type=str, default="0")

    parser.add_argument("--background-image-path", type=str,
                        default=None)
    parser.add_argument("--background-images-dir", type=str,
                        default=None)
    parser.add_argument("--background-video-path", type=str,
                        default=None)
    parser.add_argument("--background-videos-dir", type=str,
                        default="/ssd4/zhangyiyang/data/AR/generate_videos/background/results")

    parser.add_argument("--foreground-image-path", type=str,
                        default="/ssd4/zhangyiyang/background-removal/data/input/foreground/images/1.jpg")
    parser.add_argument("--foreground-images-dir", type=str,
                        default="/ssd4/zhangyiyang/background-removal/data/input/foreground/images")
    parser.add_argument("--foreground-video-path", type=str,
                        default="/ssd4/zhangyiyang/background-removal/data/input/foreground/videos/foreground.mp4")
    parser.add_argument("--foreground-videos-dir", type=str,
                        default="/ssd4/zhangyiyang/data/AR/generate_videos/foreground/left")

    parser.add_argument("--output-dir", type=str,
                        default="/ssd4/zhangyiyang/data/AR/generate_videos/results/left")

    return parser.parse_args()


def _build_model(model_type, ckpt_path):
    if model_type == 'deeplab_v3':
        return DeepLabV3(ckpt_path)
    raise ValueError("unknown model type {}".format(model_type))


def _video_generate(model,
                    in_video_cap,
                    out_video_path,
                    background_image=None,
                    background_video_cap=None,
                    background_color=DEFAULT_BACKGROUND_COLOR):
    video_width = int(in_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(in_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_size = model.set_target_size((video_height, video_width))

    if os.path.exists(out_video_path):
        os.remove(out_video_path)
    output_video_writer = cv2.VideoWriter(
        out_video_path,
        OUTPUT_CV2_FOURCC,
        in_video_cap.get(cv2.CAP_PROP_FPS),
        (target_size[0], target_size[1]),
    )

    last_background = None
    for i in tqdm(range(int(in_video_cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        flag, foreground = in_video_cap.read()
        if not flag:
            break

        # get background
        background = None
        if background_image is not None:
            background = background_image
        if background_video_cap is not None:
            flag, img = background_video_cap.read()
            if not flag:
                background = last_background
            else:
                background = img

        if background_video_cap is not None:
            last_background = background
        target_img = model.generate_image(
            foreground,
            background=background,
            background_color=background_color,
        )
        # print(foreground.shape, target_img.shape)
        output_video_writer.write(target_img)
        cv2.waitKey(1)
    in_video_cap.release()
    output_video_writer.release()
    if background_video_cap is not None:
        background_video_cap.release()


def _do_handle_single_video(video_path, model,
                            background_type, background_image,
                            background_video_cap,
                            output_dir,):
    # get output image file name & absolute path
    full_basename = os.path.basename(video_path)
    basename, ext = os.path.splitext(full_basename)
    file_name = OUTPUT_FILE_NAME_FORMAT.format(
        basename, background_type,
        int(datetime.now().timestamp()*10e6),
        OUTPUT_VIDEO_FORMAT,
    )
    # file_name = basename + background_type + OUTPUT_VIDEO_FORMAT
    output_file_path = os.path.join(output_dir, file_name)

    # generate videos
    tmp_video = './tmp1.avi'
    cmd = "ffmpeg -i {} -q:v 6 {}".format(video_path, tmp_video)
    if os.path.exists(tmp_video):
        os.remove(tmp_video)
    os.system(cmd)
    _video_generate(model,
                    cv2.VideoCapture(tmp_video),
                    output_file_path,
                    background_image=background_image,
                    background_video_cap=background_video_cap,
                    background_color=DEFAULT_BACKGROUND_COLOR)
    if os.path.exists(tmp_video):
        os.remove(tmp_video)


def _handle_single_foreground_video(video_path, model, args):
    if args.background_videos_dir is not None:
        background_type = "videos_dir"
        background_image = None
        for file_name in os.listdir(args.background_videos_dir):
            background_video_cap = cv2.VideoCapture(os.path.join(
                args.background_videos_dir, file_name,
            ))
            _do_handle_single_video(video_path, model,
                                    background_type, background_image,
                                    background_video_cap,
                                    args.output_dir)
        return

    if args.background_images_dir is not None:
        background_type = "images_dir"
        background_video_cap = None
        for file_name in os.listdir(args.background_videos_dir):
            background_image = cv2.imread(
                os.path.join(args.background_images_dir, file_name)
            )
            _do_handle_single_video(video_path, model,
                                    background_type, background_image,
                                    background_video_cap,
                                    args.output_dir)
        return

    background_type = "color"
    background_image = None
    background_video_cap = None
    if args.background_video_path is not None:
        background_video_cap = cv2.VideoCapture(args.background_video_path)
        background_type = "video"
    elif args.background_image_path is not None:
        background_image = cv2.imread(args.background_image_path)
        background_type = "image"

    _do_handle_single_video(video_path, model,
                            background_type, background_image,
                            background_video_cap,
                            args.output_dir)


def _do_handle_single_image(img_path, model,
                            background_type, background,
                            output_dir):
    # get foreground image
    foreground = cv2.imread(img_path)

    # generate target image
    target_img = model.generate_image(
        foreground,
        background=background,
        background_color=DEFAULT_BACKGROUND_COLOR,
    )

    # get output image file name & absolute path
    full_basename = os.path.basename(img_path)
    basename, ext = os.path.splitext(full_basename)
    file_name = OUTPUT_FILE_NAME_FORMAT.format(
        basename, background_type,
        int(datetime.now().timestamp()*10e6),
        OUTPUT_IMG_FORMAT,
    )
    # file_name = basename + background_type + OUTPUT_IMG_FORMAT
    output_file_path = os.path.join(output_dir, file_name)

    # write image
    cv2.imwrite(output_file_path, target_img)


def _handle_single_foreground_image(img_path, model, args):
    # 多张图片处理
    if args.background_images_dir is not None:
        background_type = "images_dir"
        for filename in os.listdir(args.background_image_dir):
            background = cv2.imread(os.path.join(
                args.background_images_dir, filename
            ))
            _do_handle_single_image(
                img_path, model, background_type, background, args.output_dir,
            )
        return

    # 处理单张图片
    if args.background_image_path is not None:
        background = cv2.imread(args.background_image_path)
        background_type = "image"
    else:
        background_type = "color"
        background = None
    _do_handle_single_image(
        img_path, model, background_type, background, args.output_dir,
    )


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    model = _build_model(args.model_type, args.ckpt_path)

    if args.foreground_videos_dir is not None:
        if not os.path.isdir(args.foreground_videos_dir):
            raise ValueError("unknown foreground videos dir {}".format(
                args.foreground_videos_dir))
        for file_name in os.listdir(args.foreground_videos_dir):
            _handle_single_foreground_video(
                os.path.join(args.foreground_videos_dir, file_name),
                model, args,
            )
    elif args.foreground_video_path is not None:
        if not os.path.exists(args.foreground_video_path):
            raise ValueError("unknown foreground video path {}".format(
                args.foreground_video_path))
        _handle_single_foreground_video(
            os.path.join(args.foreground_video_path), model, args,
        )
    elif args.foreground_images_dir is not None:
        if not os.path.isdir(args.foreground_images_dir):
            raise ValueError("unknown foreground images dir {}".format(
                args.foreground_images_dir))
        for file_name in os.listdir(args.foreground_images_dir):
            _handle_single_foreground_image(
                os.path.join(args.foreground_images_dir, file_name),
                model, args,
            )
    elif args.foreground_image_path is not None:
        if not os.path.exists(args.foreground_image_path):
            raise ValueError("unknown foreground image path {}".format(
                args.foreground_image_path
            ))
        _handle_single_foreground_image(
            args.foreground_image_path, model, args)
    else:
        raise ValueError("foreground params cannot be all None.")


if __name__ == '__main__':
    main(_parse_args())
