import cv2
import numpy as np
from . import detector as dt
import tensorflow as tf
import time

# 默认的宽度和高度
default_height = default_width = 48
# 通道数，输入时应为1
channel = 1
# 表情标签
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

'''
    初始化运算图
'''
def init():
    graph = tf.Graph()
    # 使用with新建一个graph，运行之后就会释放内存
    with graph.as_default():
        # 导入模型
        saver=tf.train.import_meta_graph('/model/cnn_emotion_classifier_g.ckpt.meta')

    return graph, saver

'''
    图像增强，归一化操作
'''
def advance_image(image):
    # 将图像从uint8的0~255表示修改为double的0~1，也就是归一化
    image = np.multiply(image, 1. / 255)
    return image

'''
    预测前的图片预处理
'''
def preprocess_image(image):
    # 调整人脸的大小
    rsz_image = cv2.resize(image, (default_height, default_width))
    # 重新生成数组 -1为任意维，事实上只有一维，也就是单通道图像
    rsz_image = np.reshape(rsz_image, [-1, default_height, default_width, channel])
    return rsz_image

'''
  预测
'''
def predict(img, graph, saver):
    # 结果数组
    res = []
    # 计时开始
    start = time.clock()
    # 对输入的图像检测人脸
    faces, gray = dt.detect_face(img)

    # 若返回了错误信息，不再检测
    if isinstance(faces, str):
        return faces

    # 复用的session，解决了多次加载模型到内存的问题。
    sess = tf.InteractiveSession(graph=graph)
    saver.restore(sess,"/model/cnn_emotion_classifier_g.ckpt")

    # 在图中获取各个节点
    # 输出节点，最后一层Logit回归的结果
    logits = graph.get_tensor_by_name('project/output/logits:0')
    # 图像输入节点
    x_input = graph.get_tensor_by_name('x_input:0')
    # dropout值输入节点
    dropout = graph.get_tensor_by_name('dropout:0')
    
    for (x, y, w, h) in faces:
        # 表情预测值
        emotion_predict = {}
        # 根据坐标得到灰度图像中的人脸
        face_img_gray = gray[y:y+h, x:x+w]
        # 图片预处理
        rsz_image = preprocess_image(face_img_gray)
        # 图片增强
        rsz_image = advance_image(rsz_image)
        # 预测结果，Logit回归的结果
        pred_logits_ = []
        # 输入模型，获取预测结果
        pred_logits_.append(sess.run(tf.nn.softmax(logits), {x_input: rsz_image, dropout: 1.0}))
        # 将结果转为python数组
        results_sum = np.sum(pred_logits_, axis=0)
        # 对预测结果与标签进行一一对应
        for i, emotion_pre in enumerate(results_sum):
            emotion_predict[emotion_labels[i]] = emotion_pre
        # 取最大值对应的标签位置
        label = np.argmax(results_sum)
        # 获取最大值对应的标签名称
        emotion = emotion_labels[int(label)]
        # 标签对应的概率
        prob = results_sum[0][int(label)]
        res.append({'emotion': emotion, 'prob': str(prob), 'position': (int(x), int(y), int(w), int(h)), 'prob_list': [str(prob) for prob in results_sum[0]] })

    # 停止计时
    end = time.clock()
    # 停止session
    sess.close()
    
    return { 'emotions': res, 'time': end-start }
