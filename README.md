# 去除/更换背景

## 1. 功能
+ 概述：去除视频/图像的背景，并根据需求替换为纯色/背景图片/背景视频。

## 2. 使用

### 2.1. 图片的去除/更换背景
+ 前景的输入形式：
  + 输入图片文件夹的路径。（优先级高）
  + 输入一张图片的路径。（优先级低）
+ 背景的输入形式：
  + 输入图片文件夹的路径。（优先级高）
  + 输入一张图片的路径。（优先级中）
  + 前面两者都不输入。（优先级低）
+ 输出图片简介：
  + 输出图片的的路径通过 `--output-dir` 设置。
  + 输出文件的名称根据输入图片的名称改变而来：在输入图片名称的基础上添加了背景的来源。
    + 如果输入文件的形式为 `{filename}.jpg`，那么输出文件的形式就是 `{filename}_{background_type}.jpg`
    + 背景来源，即上面的 `background_type`，分为图片文件夹(`images_dir`)、图片文件(`image`)、纯色(`color`)。
    + 举例：如果输入图片名称为 `image.jpg`，背景为纯色的输出图片文件名为 `image_color.jpg`。
+ 测试脚本
```shell
python src/main.py \
    --model-type deeplab_v3 \
    --ckpt-path /path/to/ckpt/dir \
    --gpu-devices 3 \
    --background-image-path /path/to/image \
    --background-images-dir /path/to/images/dir \
    --foreground-image-path /path/to/image \
    --foreground-images-dir /path/to/images/dir \
    --output-dir /path/to/output/dir
```


### 2.2. 视频的去除/更换背景
+ 前景的输入形式：
  + 输入视频文件夹的路径。（优先级高）
  + 输入视频文件的路径。（优先级低）
+ 背景的输入形式：
  + 输入视频文件夹的路径。（优先级高）
  + 输入一个视频的路径。（优先级中高）
  + 输入图片文件夹的路径。（优先级中）
  + 输入一张图片的路径。（优先级中低）
  + 前面两者都不输入。（优先级低）
+ 输出视频简介：
  + 输出图片的的路径通过 `--output-dir` 设置。
  + 输出文件的名称根据输入视频的名称改变而来：在输入视频名称的基础上添加了背景的来源。
    + 如果输入文件的形式为 `{filename}.jpg`，那么输出文件的形式就是 `{filename}_{background_type}.jpg`
    + 背景来源，即上面的 `background_type`，分为视频文件夹(`videos_dir`)、视频文件`video`)、图片文件夹(`images_dir`)、图片文件(`image`)、纯色(`color`)。
    + 举例：如果输入图片名称为 `image.jpg`，背景为纯色的输出图片文件名为 `image_color.jpg`。
+ 测试脚本
```shell
python src/main.py \
    --model-type deeplab_v3 \
    --ckpt-path /path/to/ckpt/dir \
    --gpu-devices 3 \
    --background-image-path /path/to/image \
    --background-images-dir /path/to/images/dir \
    --background-video-path /path/to/video \
    --background-videos-dir /path/to/videos/dir \
    --foreground-video-path /path/to/video \
    --foreground-videos-dir /path/to/videos/dir \
    --output-dir /path/to/output/dir
```


## 2.3. 注意事项
+ 前景是视频文件/文件夹路径的优先级高于前景是图片文件/文件夹。前景的优先级从高到低分别是：
  + `--foreground-videos-dir`
  + `--foreground-video-path`
  + `--foreground-image-path`
  + `--foreground-images-dir`
+ 背景的优先级从高到低分别是：
  + `--background-videos-dir`：仅输入视频时可用。
  + `--background-video-path`：仅输入视频时可用。
  + `--background-image-path`
  + `--background-images-dir`
  + `DEFAULT_BACKGROUND_COLOR`
+ 背景假如是纯色，则默认为白色，通过设置 `DEFAULT_BACKGROUND_COLOR` 实现。


## 3. 原理
+ 去除背景的原理：
  + 其实就是跑了一个深度学习图像分割模型DeepLabV3（不知道是不是DeepLabV3+）。
  + 模型是在VOC2012 Augmentation上训练的，其中序号0是背景。
+ 图像换背景基本原理：
  + 通过模型获取分割结果mask。
  + mask取值不为0的使用原始图像像素，取值为0的使用背景图像像素值。
  + 背景图像像素值来自背景图片或纯色。
  + 如果输入的是背景图片文件夹，则随机选择其中一张图像作为背景图片。
+ 视频换背景基本原理：
  + 如果背景是图像或纯色，就是遍历视频帧，每一帧都作为普通的图像换背景。
  + 如果背景是视频或视频文件夹，则对前景/背景视频文件分别提取帧，背景视频得到的帧作为图像换背景的背景图片。
