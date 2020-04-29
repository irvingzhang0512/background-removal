import os

import cv2
import numpy as np
import tensorflow as tf


class DeepLabV3(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'

    def __init__(self, tarball_path, img_size=None):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        graph_def = tf.GraphDef.FromString(
            open(os.path.join(tarball_path, self.FROZEN_GRAPH_NAME),
                 "rb").read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=tf_config)
        self.target_size = self.set_target_size(img_size)

    def _get_mask(self, image):
        """Runs inference on a single image.
        Args:
          image: A cv2.image object, raw input image.
        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        if self.target_size is None:
            height, width, _ = image.shape
            resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
            target_size = (int(resize_ratio * width),
                           int(resize_ratio*height))
        else:
            target_size = self.target_size
        image = image[:, :, ::-1]
        resized_image = cv2.resize(image, target_size)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})

        return resized_image[:, :, ::-1], batch_seg_map[0]

    def set_target_size(self, input_size):
        if input_size is None:
            return None
        height, width = input_size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        self.target_size = (int(resize_ratio * width),
                            int(resize_ratio*height))
        return self.target_size

    def _draw_image(self,
                    foreground, mask,
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

    def generate_image(self, image,
                       background=None,
                       background_color=(255, 255, 255)):
        foregroud, mask = self._get_mask(image)
        new_img = self._draw_image(foregroud, mask,
                                   background, background_color)
        return new_img
