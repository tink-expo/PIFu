# (Reference) https://github.com/susheelsk/image-background-removal

import os
from io import BytesIO

import numpy as np
from PIL import Image

import tensorflow as tf
import sys
import datetime

class Preprocess(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, graph_def_dir, tarball_path='xception_model'):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    graph_def_path = os.path.join(graph_def_dir, tarball_path + "/frozen_inference_graph.pb")
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(open(graph_def_path, "rb").read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.compat.v1.Session(graph=self.graph)

  @staticmethod
  def drawSegment(baseImg, matImg):
    width, height = baseImg.size
    segImg = np.zeros([height, width, 3], dtype=np.uint8)
    maskImg = np.zeros([height, width], dtype=np.uint8)
    for x in range(width):
      for y in range(height):
        color = matImg[y,x]
        (r,g,b) = baseImg.getpixel((x,y))
        if color == 0:
          segImg[y,x] = [0,0,0]
          maskImg[y,x] = 0
        else :
          segImg[y,x] = [r,g,b]
          maskImg[y,x] = 255
    return segImg, maskImg

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    start = datetime.datetime.now()

    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]

    end = datetime.datetime.now()

    diff = end - start
    print("Time taken to evaluate segmentation is : " + str(diff))

    return resized_image, seg_map

  def preprocess_image(self, original_im):
    """Inferences DeepLab model and visualizes result."""
    resized_im, seg_map = self.run(original_im)

    # vis_segmentation(resized_im, seg_map)
    seg_im, mask_im = self.drawSegment(resized_im, seg_map)
    return Image.fromarray(seg_im), Image.fromarray(mask_im)

