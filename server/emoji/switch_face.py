import cv2
from PIL import Image
import time
import numpy as np
import math

'''
    换脸
'''
def switch_face(faces, src_img, mode="cover"):
    src_img = Image.fromarray(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))

    if mode == "cover":
        for face in faces:
            x, y, w, h = face["position"]
            emoji_img = Image.open("/images/emoji/{}.png".format(face["emotion"]))
            emoji_img = emoji_img.convert("RGBA")
            new_emoji_img = emoji_img.resize((w, h), Image.LANCZOS)
            src_img.paste(new_emoji_img, (x, y), new_emoji_img)
    elif mode == "circle":
        for face in faces:
            x, y, w, h = face["position"]
            emoji_img = Image.open("/images/emoji/{}.png".format(face["emotion"]))
            emoji_img = emoji_img.convert("RGBA")
            new_emoji_img = emoji_img.resize((int(w / 5), int(h / 5)), Image.LANCZOS)
            r = math.sqrt((w / 2)**2 + (h / 2)**2)
            offset = int (r / 2)
            src_img.paste(new_emoji_img, (x, y - offset), new_emoji_img)
            src_img.paste(new_emoji_img, (x + w, y - offset), new_emoji_img)
            # src_img.paste(new_emoji_img, (x, y + h), new_emoji_img)
            # src_img.paste(new_emoji_img, (x + w, y + h), new_emoji_img)
            src_img.paste(new_emoji_img, (int(x + w / 2 - r), int(y + h / 2) - offset), new_emoji_img)
            src_img.paste(new_emoji_img, (int(x + w / 2 + r), int(y + h / 2) - offset), new_emoji_img)
            src_img.paste(new_emoji_img, (int(x + w / 2), int(y + h / 2 - r) - offset), new_emoji_img)
            # src_img.paste(new_emoji_img, (int(x + w / 2), int(y + h / 2 + r)), new_emoji_img)

    nowTime = (int(round(time.time() * 1000)))
    res_img_path = "/static/emoji-result-{}.jpg".format(nowTime)
    src_img.save("/server{}".format(res_img_path))

    return res_img_path

