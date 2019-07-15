from .load import DeepLabModel
import numpy as np
from PIL import Image
import cv2
import time

# 模型存放位置
model_path = '/model/mobilenetv2/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'

'''
    模型初始化
'''
def init():
    return DeepLabModel(model_path)

'''
    模型预测
'''
def predict(img, bg_img, model, is_background):
    seg_map, cost_time = model.run(img)
    res_img_path = segment(seg_map, img, bg_img, is_background)
    return { "res_img_path": res_img_path, "time": cost_time, "is_background": is_background }

'''
    根据预测结果进行分割
'''
def segment(seg_map, src_img, bg_img, is_background):    
    # 生成分割图像
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    # 只留下标签为人的分割图像
    seg_image = np.where(seg_image!=192,0,192)

    height, width, __ = seg_image.shape
    src_height, src_width, ___ = src_img.shape
    # 根据分割图像的大小对现有图像进行调整
    src_img = cv2.resize(src_img, (width, height))
    if is_background == "true":
        bg_height, bg_width, ____ = bg_img.shape
        bg_img = cv2.resize(bg_img, (width, height))

    for ia in range(0,height):
        for ib in range(0,width):
            # 根据分割图像，合并背景和原图
            # 带背景的图是把原图写到背景上
            # 不带背景的图是把原图中非背景的地方设为黑色，便于转换为透明
            if (seg_image[ia][ib] == np.array([0,0,0])).all():
                src_img[ia][ib] = [0,0,0]
            elif is_background == "true":
                bg_img[ia][ib] = src_img[ia][ib]
    
    nowTime = (int(round(time.time() * 1000)))

    # # 转回原大小
    # src_img = cv2.resize(src_img, (src_width, src_height))
    # 转为PIL格式
    src_img = Image.fromarray(cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB))

    if is_background == "true":
        bg_img = cv2.resize(bg_img, (src_width, src_height))
        bg_img = Image.fromarray(cv2.cvtColor(bg_img,cv2.COLOR_BGR2RGB))
        res_img_path = "/static/matting-result-{}.jpg".format(nowTime)
        bg_img.save("/server{}".format(res_img_path))
    else:
        res_img_path = "/static/matting-result-{}.png".format(nowTime)
        transparent_img = transparent_back(src_img)
        transparent_img = transparent_img.resize((src_width, src_height))
        transparent_img.save("/server{}".format(res_img_path))

    return res_img_path


'''
    创建分割图像
'''
def create_pascal_label_colormap():
    # 从256色中选取颜色值
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    # 每遇到一种标签就进行映射，分别映射R、G、B三个通道
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    # 返回分割图像
    return colormap

'''
    由标签转化为带有颜色的分割图像
'''
def label_to_color_image(label):
    # 若输入的维数不正确，抛错
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    # 创建分割图像
    colormap = create_pascal_label_colormap()

    # 若标签的个数大于分割图像颜色的个数，抛错
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    # 返回分割图像
    return colormap[label]

'''
    若不需要背景图，则以第一个像素为准设为透明
'''
def transparent_back(img):
    img = img.convert('RGBA')
    L, H = img.size
    # 透明颜色
    color_0 = np.array([0, 0, 0, 255])
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if (color_1 == color_0).all():
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,color_1)
    return img



    




