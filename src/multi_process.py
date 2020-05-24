import time
import cv2
import os
import argparse
import subprocess

import numpy as np
from datetime import datetime
from deeplab_v3 import DeepLabV3
from multiprocessing import Pool


DEFAULT_BACKGROUND_COLOR = (255, 255, 255)
OUTPUT_IMG_FORMAT = ".png"
OUTPUT_VIDEO_FORMAT = ".avi"
OUTPUT_CV2_FOURCC = cv2.VideoWriter_fourcc(* 'XVID')
# origin_name/model/background/time/pid/ext
OUTPUT_FILE_NAME_FORMAT = "{}_{}_{}_{}_{}{}"
OUTPUT_FILE_NAME_TIME_FORMAT = "%Y%m%d%H%M%S"
OUTPUT_VIDEO_FPS = 25


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="deeplab_v3")
    parser.add_argument("--ckpt-path", type=str,
                        default="./data/deeplab_v3/xception")
    parser.add_argument("--gpu-devices", type=str, default="0")
    parser.add_argument("--pool-size", type=int, default=10)

    parser.add_argument("--background-image-path", type=str,
                        default=None)
    parser.add_argument("--background-images-dir", type=str,
                        default=None)
    parser.add_argument("--background-video-path", type=str,
                        default=None)
    parser.add_argument("--background-videos-dir", type=str,
                        default=None)

    parser.add_argument("--foreground-image-path", type=str,
                        default=None)
    parser.add_argument("--foreground-images-dir", type=str,
                        default=None)
    parser.add_argument("--foreground-video-path", type=str,
                        default="./data/input/raw/wbj2.mp4")
    parser.add_argument("--foreground-videos-dir", type=str,
                        default='./data/input/clips/left')

    parser.add_argument("--output-dir", type=str,
                        default="./data/output/left")

    return parser.parse_args()


def _build_model(model_type, ckpt_path):
    if model_type == 'deeplab_v3':
        model = DeepLabV3(ckpt_path)
        return model, model.INPUT_SIZE
    raise ValueError("unknown model type {}".format(model_type))


def _draw_image(foreground, mask,
                background=None, background_color=(255, 255, 255)):
    height, width = foreground.shape[:2]
    if background is not None:
        background = cv2.resize(
            background, (width, height))
    dummy_img = np.zeros([height, width, 3], dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            is_foreground = (mask[i, j] != 0)
            if is_foreground:
                dummy_img[i, j] = foreground[i, j]
            else:
                dummy_img[i, j] = background[i, j] \
                    if background is not None \
                    else background_color
    return dummy_img


def _do_generate_image(fgimage, mask, model,
                       bgimage, bgcolor,):
    if mask is None:
        origi_img, cur_mask, target_img = model.generate_image(
            fgimage,
            background=bgimage,
            background_color=bgcolor,
            mask=None,
        )
        return origi_img, cur_mask, target_img
    else:
        h, w = mask.shape[:2]
        fgimage = cv2.resize(fgimage, (w, h))
        target_img = _draw_image(
            fgimage, mask,
            background=bgimage,
            background_color=bgcolor,
        )
        return fgimage, mask, target_img


def _video_generate_v2(fgs, masks,
                       out_video_path,
                       background_image=None,
                       background_video_cap=None,
                       background_color=DEFAULT_BACKGROUND_COLOR):
    if os.path.exists(out_video_path):
        os.remove(out_video_path)
    output_video_writer = cv2.VideoWriter(
        out_video_path,
        OUTPUT_CV2_FOURCC,
        OUTPUT_VIDEO_FPS,
        (masks[0].shape[1], masks[0].shape[0]),
    )

    last_background = None
    for fgimage, mask in zip(fgs, masks):
        # get background
        bgimage = None
        if background_image is not None:
            bgimage = background_image
        if background_video_cap is not None:
            flag, bgimage = background_video_cap.read()
            if not flag:
                bgimage = last_background
            else:
                last_background = bgimage

        _, _, target_img = _do_generate_image(
            fgimage, mask, None, bgimage, background_color
        )
        output_video_writer.write(target_img)
    output_video_writer.release()
    if background_video_cap is not None:
        background_video_cap.release()

    return fgs, masks


def _video_generate_v1(model,
                       in_video_cap,
                       out_video_path,
                       background_image=None,
                       background_video_cap=None,
                       background_color=DEFAULT_BACKGROUND_COLOR,
                       generate_mask=False):
    video_width = int(in_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(in_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_size = model.set_target_size((video_height, video_width))

    if os.path.exists(out_video_path):
        os.remove(out_video_path)
    output_video_writer = cv2.VideoWriter(
        out_video_path,
        OUTPUT_CV2_FOURCC,
        OUTPUT_VIDEO_FPS,
        (target_size[0], target_size[1]),
    )

    last_background = None
    if generate_mask:
        masks = []
        fgs = []
    else:
        masks = None
        fgs = None
    for i in range(int(in_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        flag, fgimage = in_video_cap.read()
        if not flag:
            break

        # get background
        bgimage = None
        if background_image is not None:
            bgimage = background_image
        elif background_video_cap is not None:
            flag, img = background_video_cap.read()
            if not flag:
                bgimage = last_background
            else:
                bgimage = img
                last_background = bgimage

        origi_img, cur_mask, target_img = _do_generate_image(
            fgimage, None, model, bgimage, background_color
        )
        if generate_mask:
            masks.append(cur_mask)
            fgs.append(origi_img)
        output_video_writer.write(target_img)
        cv2.waitKey(1)
    in_video_cap.release()
    output_video_writer.release()
    if background_video_cap is not None:
        background_video_cap.release()

    return fgs, masks


def _do_handle_single_fgvideo(video_path, model, model_type,
                              fgs, masks,
                              background_type,
                              background_image,
                              background_video_path,
                              output_dir,):
    # get output image file name & absolute path
    full_basename = os.path.basename(video_path)
    basename, ext = os.path.splitext(full_basename)
    file_name = OUTPUT_FILE_NAME_FORMAT.format(
        basename, model_type, background_type,
        datetime.now().strftime(OUTPUT_FILE_NAME_TIME_FORMAT),
        os.getpid(),
        OUTPUT_VIDEO_FORMAT,
    )
    # file_name = basename + background_type + OUTPUT_VIDEO_FORMAT
    output_file_path = os.path.join(output_dir, file_name)

    print('start generating {}'.format(output_file_path))
    # generate videos
    tmp_video = './{}.avi'.format(datetime.now().timestamp())
    cmd = "ffmpeg -i {} -q:v 6 {}".format(video_path, tmp_video)
    if os.path.exists(tmp_video):
        os.remove(tmp_video)
    subprocess.call(
        cmd,
        shell=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if masks is None:
        background_video_cap = None
        if background_video_path is not None:
            background_video_cap = cv2.VideoCapture(background_video_path)
        fgs, masks = _video_generate_v1(
            model,
            cv2.VideoCapture(video_path),
            output_file_path,
            background_image=background_image,
            background_video_cap=background_video_cap,
            background_color=DEFAULT_BACKGROUND_COLOR,
            generate_mask=True)
    else:
        background_video_cap = None
        if background_video_path is not None:
            background_video_cap = cv2.VideoCapture(background_video_path)
        fpgs, masks = _video_generate_v2(
            fgs, masks,
            output_file_path,
            background_image=background_image,
            background_video_cap=background_video_cap,
            background_color=DEFAULT_BACKGROUND_COLOR,
        )
    print('finish generating {}'.format(output_file_path))
    if os.path.exists(tmp_video):
        os.remove(tmp_video)
    return fgs, masks


def _multi_processing_single_video(video_path,
                                   model, model_type,
                                   output_dir,
                                   bgimages_dir=None,
                                   bgvideos_dir=None,
                                   pool_size=10,
                                   ):
    bgimages_paths = None
    bgvideos_paths = None
    left_sample_cnt = 0
    if bgimages_paths is not None:
        background_type = "images_dir"
        bgimages_paths = [os.path.join(bgimages_dir, f)
                          for f in os.listdir(bgimages_dir)]
        left_sample_cnt = len(bgimages_paths) - 1
    else:
        background_type = "videos_dir"
        bgvideos_paths = [os.path.join(bgvideos_dir, f)
                          for f in os.listdir(bgvideos_dir)]
        left_sample_cnt = len(bgvideos_paths) - 1

    fgs, masks = _do_handle_single_fgvideo(
        video_path, model, model_type,
        None, None,
        background_type,
        cv2.imread(bgimages_paths[0]) if bgimages_paths else None,
        bgvideos_paths[0] if bgvideos_paths else None,
        output_dir
    )

    def _fail(e):
        print(e)

    # multi process
    with Pool(pool_size) as pool:
        for i in range(left_sample_cnt):
            pool.apply_async(
                _do_handle_single_fgvideo,
                args=(
                    video_path, None, model_type,
                    fgs, masks,
                    background_type,
                    cv2.imread(
                        bgimages_paths[i+1]) if bgimages_paths else None,
                    bgvideos_paths[i+1] if bgvideos_paths else None,
                    output_dir
                ),
                error_callback=_fail,
            )
        pool.close()
        pool.join()
        print("finish generating {} videos".format(video_path))

    # # single process
    # for i in range(left_sample_cnt):
    #     _do_handle_single_fgvideo(
    #         video_path, None, model_type,
    #         fgs, masks,
    #         background_type,
    #         cv2.imread(bgimages_paths[i + 1]
    #                    ) if bgimages_paths is not None else None,
    #         bgvideos_paths[i+1] if bgvideos_paths is not None else None,
    #         output_dir
    #     )


def _handle_single_fgvideo(video_path, model, args):
    if args.background_images_dir is not None \
            or args.background_videos_dir is not None:
        _multi_processing_single_video(
            video_path,
            model, os.path.basename(args.ckpt_path),
            args.output_dir,
            args.background_images_dir,
            args.background_videos_dir,
            args.pool_size,
        )
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

    _video_generate_v1(video_path, model,
                       None, None,
                       background_type,
                       background_image,
                       background_video_cap,
                       args.output_dir,)


def _handle_single_fgbgimage(img_path, model, model_type,
                             mask,
                             background_type, bgimage,
                             output_dir,):
    # 处理单张前景图片 + 单张背景图片/颜色

    # get foreground image
    fgimage = cv2.imread(img_path)

    # generate target image
    _, cur_mask, target_img = _do_generate_image(
        fgimage, mask, model,
        bgimage, DEFAULT_BACKGROUND_COLOR,
    )

    # get output image file name & absolute path
    full_basename = os.path.basename(img_path)
    basename, ext = os.path.splitext(full_basename)
    file_name = OUTPUT_FILE_NAME_FORMAT.format(
        basename, model_type, background_type,
        datetime.now().strftime(OUTPUT_FILE_NAME_TIME_FORMAT),
        os.getpid(),
        OUTPUT_IMG_FORMAT,
    )
    # file_name = basename + background_type + OUTPUT_IMG_FORMAT
    output_file_path = os.path.join(output_dir, file_name)

    # write image
    cv2.imwrite(output_file_path, target_img)

    return cur_mask


def _handle_single_fgimage(img_path, model, args):
    # 处理单张前景图片的各种情况

    model_type = os.path.basename(args.ckpt_path)

    # 多张背景图片处理
    if args.background_images_dir is not None:
        background_type = "images_dir"
        mask = None
        for filename in os.listdir(args.background_image_dir):
            background = cv2.imread(os.path.join(
                args.background_images_dir, filename
            ))
            mask = _handle_single_fgbgimage(
                img_path, model, model_type, mask,
                background_type, background,
                args.output_dir,
            )
        return

    # 处理单张背景图片
    if args.background_image_path is not None:
        background = cv2.imread(args.background_image_path)
        background_type = "image"
    else:
        background_type = "color"
        background = None
    _handle_single_fgbgimage(
        img_path, model, model_type, None,
        background_type, background,
        args.output_dir,
    )


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    model, input_size = _build_model(args.model_type, args.ckpt_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.foreground_videos_dir is not None:
        if not os.path.isdir(args.foreground_videos_dir):
            raise ValueError("unknown foreground videos dir {}".format(
                args.foreground_videos_dir))
        t1 = time.time()
        for file_name in os.listdir(args.foreground_videos_dir):
            _handle_single_fgvideo(
                os.path.join(args.foreground_videos_dir, file_name),
                model, args,
            )
        print(time.time() - t1)
    elif args.foreground_video_path is not None:
        if not os.path.exists(args.foreground_video_path):
            raise ValueError("unknown foreground video path {}".format(
                args.foreground_video_path))
        _handle_single_fgvideo(
            os.path.join(args.foreground_video_path), model, args,
        )
    elif args.foreground_images_dir is not None:
        if not os.path.isdir(args.foreground_images_dir):
            raise ValueError("unknown foreground images dir {}".format(
                args.foreground_images_dir))
        for file_name in os.listdir(args.foreground_images_dir):
            _handle_single_fgimage(
                os.path.join(args.foreground_images_dir, file_name),
                model, args,
            )
    elif args.foreground_image_path is not None:
        if not os.path.exists(args.foreground_image_path):
            raise ValueError("unknown foreground image path {}".format(
                args.foreground_image_path
            ))
        _handle_single_fgimage(
            args.foreground_image_path, model, args)
    else:
        raise ValueError("foreground params cannot be all None.")


if __name__ == '__main__':
    main(_parse_args())
