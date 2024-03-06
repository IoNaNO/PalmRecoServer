from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
import os
import base64
from PIL import Image
import io
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import edcc

TRAIN_SET = 'train'
TEST_SET = 'test'
app = Flask(__name__, static_url_path = "")
app.config['TRAIN_SET'] = TRAIN_SET
app.config['TEST_SET'] = TEST_SET
CORS(app, resources=r'/*')
auth = HTTPBasicAuth()

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

conf = edcc.EncoderConfig(29, 5, 5 ,10)
encoder = edcc.create_encoder(conf)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        ax = hand_landmarks[5].x
        ay = hand_landmarks[5].y
        bx = hand_landmarks[9].x
        by = hand_landmarks[9].y
        cx = hand_landmarks[13].x
        cy = hand_landmarks[13].y
        dx = hand_landmarks[17].x
        dy = hand_landmarks[17].y

        # # Draw the hand landmarks.
        # hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        # hand_landmarks_proto.landmark.extend([
        #     landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        # ])
        # solutions.drawing_utils.draw_landmarks(
        #     annotated_image,
        #     hand_landmarks_proto,
        #     solutions.hands.HAND_CONNECTIONS,
        #     solutions.drawing_styles.get_default_hand_landmarks_style(),
        #     solutions.drawing_styles.get_default_hand_connections_style())

    # # Get the top left corner of the detected hand's bounding box.
    # height, width, _ = annotated_image.shape
    # x_coordinates = [landmark.x for landmark in hand_landmarks]
    # y_coordinates = [landmark.y for landmark in hand_landmarks]
    # text_x = int(min(x_coordinates) * width)
    # text_y = int(min(y_coordinates) * height) - MARGIN

    # # Draw handedness (left or right hand) on the image.
    # cv2.putText(annotated_image, f"{handedness[0].category_name}",
    #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
    #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    # return annotated_image
    return ax, ay, bx, by, cx, cy, dx, dy

@app.route('/recognize', methods=['POST'])
def recognize():
    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file part in the request.'}), 400

    # file = request.files['file']
    # if file.filename == '':
    #     return jsonify({'error': 'No selected file.'}), 400
        
    try:
        # data = request.get_json()
        # img_data = data['file']
        img_data = request.form.get('file')
    except Exception as e:
        return jsonify({'code': 1}), 401

    if img_data:
        # filename = secure_filename(file.filename)
        save_path = os.path.join('./Users', app.config['TEST_SET'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # img_data = img_data.replace(' ', '+')
        img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data, altchars=None, validate=False)
        img = Image.open(io.BytesIO(img_bytes))
        img.save(os.path.join(save_path, 'temp.jpg'))
        # file.save(os.path.join(save_path, 'temp.jpg'))
        
#        img = mp.Image.create_from_file(os.path.join(save_path, 'temp.jpg'))
#        detection_result = detector.detect(img)
#        ax, ay, bx, by, cx, cy, dx, dy = draw_landmarks_on_image(img.numpy_view(), detection_result)
        
        img = cv2.imread(os.path.join(save_path, 'temp.jpg'), 1)
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(save_path, 'temp.jpg'), img)
        img_ = mp.Image.create_from_file(os.path.join(save_path, 'temp.jpg'))
        detection_result = detector.detect(img_)
        ax, ay, bx, by, cx, cy, dx, dy = draw_landmarks_on_image(img_.numpy_view(), detection_result)
        img = cv2.imread(os.path.join(save_path, 'temp.jpg'), 0)
        h, w = img.shape
        v1 = np.array([(0.67 * ax + 0.33 * bx) * w,
                       (0.67 * ay + 0.33 * by) * h])
        v2 = np.array([(0.33 * cx + 0.67 * dx) * w,
                       (0.33 * cy + 0.67 * dy) * h])
        theta = np.arctan2(v2[1] - v1[1], v2[0] - v1[0]) * 180 / np.pi
        R = cv2.getRotationMatrix2D((int(v2[0]), int(v2[1])), theta, 1)
        rotated_img = cv2.warpAffine(img, R, (w, h))
        v1 = (R[:,:2] @ v1 + R[:,-1]).astype(np.int32)
        v2 = (R[:,:2] @ v2 + R[:,-1]).astype(np.int32)
        ux = int(v1[0])
        uy = int(v1[1])
        lx = int(v2[0])
        ly = int(v2[1] + v2[0] - v1[0])
        ROI_img = rotated_img[uy:ly,ux:lx]
        cv2.imwrite(os.path.join(save_path, 'temp_ROI.jpg'), ROI_img)
    
        for root, dirs, files in os.walk('./Users'):
            if root == './Users':
                for dir in dirs:
                    if dir != 'test':
                        test_path = os.path.join('./Users', dir)
                        test_path = os.path.join(test_path, app.config['TRAIN_SET'] + '/left')
                        for i in range(2):
                            one_palmprint_code = encoder.encode_using_file(os.path.join(test_path, f'left_ROI_{i}.jpg'))
                            another_palmprint_code = encoder.encode_using_file(os.path.join(save_path, 'temp_ROI.jpg'))
                            if one_palmprint_code.compare_to(another_palmprint_code) >= 0.12:
                                return jsonify({'code': 0, 'result': dir})

                        test_path = os.path.join('./Users', dir)
                        test_path = os.path.join(test_path, app.config['TRAIN_SET'] + '/right')
                        for i in range(2):
                            one_palmprint_code = encoder.encode_using_file(os.path.join(test_path, f'right_ROI_{i}.jpg'))
                            another_palmprint_code = encoder.encode_using_file(os.path.join(save_path, 'temp_ROI.jpg'))
                            if one_palmprint_code.compare_to(another_palmprint_code) >= 0.12:
                                return jsonify({'code': 0, 'result': dir})


        return jsonify({'code': 0, 'result': 'None'})
    return jsonify({'code': 1}), 400

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        print(data.keys())
        user_name = data['username']
        left_images = data['left_images']
        right_images = data['right_images']
    except Exception as e:
        return jsonify({'code': 1}), 401

    save_path = os.path.join('./Users/' + user_name, app.config['TRAIN_SET'])

    left_path = os.path.join(save_path, 'left')
    if not os.path.exists(left_path):
        os.makedirs(left_path)
    for i, img_data in enumerate(left_images):
        img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data, altchars=None, validate=False)
        img = Image.open(io.BytesIO(img_bytes))
        img.save(os.path.join(left_path, f'left_image_{i}.jpg'))

        # img = mp.Image.create_from_file(os.path.join(left_path, f'left_image_{i}.jpg'))
        # detection_result = detector.detect(img)
        # ax, ay, bx, by, cx, cy, dx, dy = draw_landmarks_on_image(img.numpy_view(), detection_result)
        
        img = cv2.imread(os.path.join(left_path, f'left_image_{i}.jpg'), 1)
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(left_path, f'left_image_{i}.jpg'), img)
        img_ = mp.Image.create_from_file(os.path.join(left_path, f'left_image_{i}.jpg'))
        detection_result = detector.detect(img_)
        ax, ay, bx, by, cx, cy, dx, dy = draw_landmarks_on_image(img_.numpy_view(), detection_result)
        img = cv2.imread(os.path.join(left_path, f'left_image_{i}.jpg'), 0)
        h, w = img.shape
        v1 = np.array([(0.67 * ax + 0.33 * bx) * w,
                       (0.67 * ay + 0.33 * by) * h])
        v2 = np.array([(0.33 * cx + 0.67 * dx) * w,
                       (0.33 * cy + 0.67 * dy) * h])
        theta = np.arctan2(v2[1] - v1[1], v2[0] - v1[0]) * 180 / np.pi
        R = cv2.getRotationMatrix2D((int(v2[0]), int(v2[1])), theta, 1)
        rotated_img = cv2.warpAffine(img, R, (w, h))
        v1 = (R[:,:2] @ v1 + R[:,-1]).astype(np.int32)
        v2 = (R[:,:2] @ v2 + R[:,-1]).astype(np.int32)
        ux = int(v1[0])
        uy = int(v1[1])
        lx = int(v2[0])
        ly = int(v2[1] + v2[0] - v1[0])
        print(uy, ly)
        print(ux, lx)
        ROI_img = rotated_img[uy:ly,ux:lx]
        cv2.imwrite(os.path.join(left_path, f'left_ROI_{i}.jpg'), ROI_img)

    right_path = os.path.join(save_path, 'right')
    if not os.path.exists(right_path):
        os.makedirs(right_path)
    for i, img_data in enumerate(right_images):
        img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data, altchars=None, validate=False)
        img = Image.open(io.BytesIO(img_bytes))
        img.save(os.path.join(right_path, f'right_image_{i}.jpg'))

        # img = mp.Image.create_from_file(os.path.join(right_path, f'right_image_{i}.jpg'))
        # detection_result = detector.detect(img)
        # ax, ay, bx, by, cx, cy, dx, dy = draw_landmarks_on_image(img.numpy_view(), detection_result)
        
        img = cv2.imread(os.path.join(right_path, f'right_image_{i}.jpg'), 1)
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(right_path, f'right_image_{i}.jpg'), img)
        img_ = mp.Image.create_from_file(os.path.join(right_path, f'right_image_{i}.jpg'))
        detection_result = detector.detect(img_)
        ax, ay, bx, by, cx, cy, dx, dy = draw_landmarks_on_image(img_.numpy_view(), detection_result)
        img = cv2.imread(os.path.join(right_path, f'right_image_{i}.jpg'), 0)
        h, w = img.shape
        v1 = np.array([(0.67 * ax + 0.33 * bx) * w,
                       (0.67 * ay + 0.33 * by) * h])
        v2 = np.array([(0.33 * cx + 0.67 * dx) * w,
                       (0.33 * cy + 0.67 * dy) * h])
        theta = np.arctan2(v2[1] - v1[1], v2[0] - v1[0]) * 180 / np.pi
        R = cv2.getRotationMatrix2D((int(v2[0]), int(v2[1])), theta, 1)
        rotated_img = cv2.warpAffine(img, R, (w, h))
        v1 = (R[:,:2] @ v1 + R[:,-1]).astype(np.int32)
        v2 = (R[:,:2] @ v2 + R[:,-1]).astype(np.int32)
        ux = int(v1[0])
        uy = int(v1[1])
        lx = int(v2[0])
        ly = int(v2[1] + v2[0] - v1[0])
        print(uy, ly)
        print(ux, lx)
        ROI_img = rotated_img[uy:ly,ux:lx]
        cv2.imwrite(os.path.join(right_path, f'right_ROI_{i}.jpg'), ROI_img)


    return jsonify({'code': 0})

if __name__ == '__main__':
    context = ('cert.pem', 'key.pem')
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=context)
