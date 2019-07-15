import cv2
from PIL import Image
import time

'''
    换脸
'''
def switch_face(faces, src_img):
    src_img = Image.fromarray(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))

    for face in faces:
        emoji_img = Image.open("/images/emoji/{}.png".format(face["emotion"]))
        emoji_img = emoji_img.convert("RGBA")
        new_emoji_img = emoji_img.resize((face["position"][2], face["position"][3]), Image.LANCZOS)
        src_img.paste(new_emoji_img, (face["position"][0], face["position"][1]), new_emoji_img)

    nowTime = (int(round(time.time() * 1000)))
    res_img_path = "/static/emoji-result-{}.jpg".format(nowTime)
    src_img.save("/server{}".format(res_img_path))

    return res_img_path

