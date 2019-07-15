import os
import tensorflow as tf
import tarfile
from PIL import Image
import time
import cv2
import numpy as np

# 输入输出的节点名称
INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
# 图像输入大小
INPUT_SIZE = 513
# 模型文件名称
FROZEN_GRAPH_NAME = 'frozen_inference_graph'

'''
    DeepLab语义分割模型类
'''
class DeepLabModel(object):
    # 类构造函数
    def __init__(self, tarball_path):
        # 加载运算图
        self.graph = tf.Graph()
        # 运算图定义
        graph_def = None
        
        # 读取模型压缩文件
        tar_file = tarfile.open(tarball_path)
        # 在压缩文件中寻找frozen_inference_graph这个模型文件
        # 解压出来，读入到运算图定义中
        for tar_info in tar_file.getmembers():
            if FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        # 结束读取
        tar_file.close()

        # 若读取出来的运算图定义为空，说明找不到模型文件
        if graph_def is None:
            raise RuntimeError('找不到模型文件')

        # 将该运算图读入到tensorflow中
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')


    '''
        运行预测服务
    '''
    def run(self, image):
         # 复用的session，解决了多次加载模型到内存的问题。
        self.sess = tf.InteractiveSession(graph=self.graph)
        # 计时开始
        start = time.clock()
        # 读入图像的宽高
        height, width, __ = image.shape
        # 模型缩放到指定的输入大小的缩放率
        resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
        # 根据缩放率算出具体的缩放长宽
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # 根据缩放长宽缩放图片
        resized_image = cv2.resize(image, target_size)
        # 将图片输入到神经网络中进行预测
        batch_seg_map = self.sess.run(
            OUTPUT_TENSOR_NAME,
            feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        # 得到预测结果
        seg_map = batch_seg_map[0]
        # 停止计时
        end = time.clock()
        # 停止session
        self.sess.close()
        return seg_map, end - start