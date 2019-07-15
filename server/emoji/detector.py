# coding:utf-8
import cv2
from PIL import Image
import numpy as np
import io

'''
    图像旋转
'''
def rotate(converted_img, degree):
    # 开始旋转
    (h, w) = converted_img.shape[:2]
    (center_x, center_y) = (w // 2, h // 2)    
    # 获取旋转矩阵
    rotate_matrix = cv2.getRotationMatrix2D((center_x, center_y), degree, 1.0)
    # 计算旋转角度的cos和sin
    cos = np.abs(rotate_matrix[0, 0])
    sin = np.abs(rotate_matrix[0, 1])
    # 根据cos和sin计算新的宽度和高度
    new_width = int((h * sin) + (w * cos))
    new_height = int((h * cos) + (w * sin))
    rotate_matrix[0, 2] += (new_width / 2) - center_x
    rotate_matrix[1, 2] += (new_height / 2) - center_y
    # 旋转
    rotated_img = cv2.warpAffine(converted_img, rotate_matrix, (new_width, new_height))
    return rotated_img

'''
    上传图像预处理
'''
def img_convert(req_img, is_front):
    # 根据前后摄旋转图片方向
    if is_front == "true":
        degree = 90
    elif is_front == "false":
        degree = -90
    else: 
        degree = 0
    
    # 转码为opencv格式
    in_memory_file = io.BytesIO()
    req_img.save(in_memory_file)
    nparr = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    converted_img = cv2.imdecode(nparr, color_image_flag)
    # 图像旋转
    if degree != 0:
        rotated_img = rotate(converted_img, degree)
    else:
        rotated_img = converted_img
    
    return rotated_img

'''
  HAAR特征检测人脸
  
  用opencv的方式（HAAR特征）检测人脸。
'''
def detect_face(image):
    # 获取训练好的人脸参数数据，此处引用GitHub上的opencv库中的默认值
    face_cascade = cv2.CascadeClassifier(r'/model/haarcascade_frontalface_default.xml')

    if isinstance(image, str):
        # 读取图片，并处理成灰度图
        image = cv2.imread(image)

    # 未读取到图片，返回
    if image is None:
        return "提示：未读取到图片", None
    
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # haar模型检测人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.15,
        minNeighbors = 3,
        minSize = (85, 85),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    # 未检测到人脸，返回
    if len(faces) <= 0:
        return "提示：未检测到人脸", gray
    
    return faces, gray