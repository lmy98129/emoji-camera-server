from flask import Flask, request, jsonify
import json

import emoji.main as emoji
import emoji.detector as dt
import emoji.switch_face as sf
import matting.main as matting

app = Flask(__name__)

emoji_graph = None
emoji_saver = None
matting_model = None

def app_init():
    graph, saver = emoji.init()
    model = matting.init()
    return graph, saver, model

@app.route('/emoji', methods=['POST'])
def emoji_server():
    img = request.files['upload']
    is_front = request.form.get('is_front')

    if img is None:
        return jsonify({ 'success': False, 'res': "未接收到图片" })

    img = dt.img_convert(img, is_front)

    if Flask.got_first_request:
        emoji_graph, emoji_saver, matting_model = app_init()
    
    res = emoji.predict(img, emoji_graph, emoji_saver)

    if isinstance(res, str):
        return jsonify({ 'success': False, 'res': res })
    else:
        res_img_path = sf.switch_face(res["emotions"], img)
        res['img_path'] = res_img_path
        return jsonify({ 'success': True, 'res': res })

@app.route('/matting', methods=['POST'])
def matting_server():
    img = request.files['upload']
    is_front = request.form.get('is_front')
    is_background = request.form.get('is_background')

    if img is None:
        return jsonify({ 'success': False, 'res': "未接收到图片" })
    if is_background == 'true':
        background = request.files['background']
        bg_img = dt.img_convert(background, "")
        if background is None:
            return jsonify({ 'success': False, 'res': "未接收到背景" })
    else:
        bg_img = None

    img = dt.img_convert(img, is_front)

    if Flask.got_first_request:
        emoji_graph, emoji_saver, matting_model = app_init()
    
    res = matting.predict(img, bg_img, matting_model, is_background)

    return jsonify({ 'success': True, 'res': res })


if __name__ == '__main__':
    app.run(debug=True)