from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
import os
import base64
from PIL import Image
import io
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import edcc
import time

RIGHT = 0
LEFT  = 1
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

MARGIN = 10
FONT_SIZE=1
FONT_THICKNESS=1
HANDEDNESS_TEXT_COLOR=(0, 0, 255)

# generate annotated image
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    # return annotated_image
    return annotated_image

# generate ROI image
def get_ROI_image(gray_image, detection_result):
    hand_landmarks=detection_result.hand_landmarks[0]
    # pick key points
    x5, y5 = hand_landmarks[5].x, hand_landmarks[5].y
    x17, y17 = hand_landmarks[17].x, hand_landmarks[17].y

    # get the coordinates
    h, w = gray_image.shape
    x5, y5 = int(x5 * w), int(y5 * h)
    x17, y17 = int(x17 * w), int(y17 * h)

    # middle point
    cx, cy = (x5 + x17) // 2, (y5 + y17) // 2

    # angle of rotation
    if detection_result.handedness[0][0].index == LEFT:
        angle = np.arctan2(y17 - y5, x17 - x5) * (180 / np.pi)
    else:
        angle = np.arctan2(y5 - y17, x5 - x17) * (180 / np.pi)

    # new sqaure side length
    length=int(np.hypot(x17-x5, y17-y5))
    
    # rotate matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    rotated_img = cv2.warpAffine(gray_image, M, (w, h))
    # cv2.imwrite('rotated.jpg', rotated_img)

    # ROI
    start_x = cx - length // 2
    start_y = cy
    ROI_img = rotated_img[start_y:start_y + length, start_x:start_x + length]

    # normalization
    ROI_img = cv2.resize(ROI_img, (256, 256))

    return ROI_img

# decode base64 image
def decode_base64_image(img_data):
    img_data = img_data.split(",")[1]
    img_bytes = base64.b64decode(img_data, altchars=None, validate=False)
    img = Image.open(io.BytesIO(img_bytes))
    return img

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        img_data = request.form.get('file')
    except Exception as e:
        return jsonify({'code': 1}), 401

    if img_data:
        st_time=time.time()
        save_path = os.path.join('./Users', app.config['TEST_SET'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # decode base64 image
        img = decode_base64_image(img_data)
        img.save(os.path.join(save_path, 'temp.jpg'))
        
        # pre process image
        # img = cv2.imread(os.path.join(save_path, 'temp.jpg'), 1)
        # img = cv2.transpose(img)
        # img = cv2.flip(img, 1)
        # cv2.imwrite(os.path.join(save_path, 'temp.jpg'), img)
        # rgb_frame = mp.Image.create_from_file(os.path.join(save_path, 'temp.jpg'))
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))
        
        # detect landmarks
        detection_result = detector.detect(rgb_frame)
        ed_time = time.time()
        print('detection time:', ed_time()-st_time)
        # print(detection_result)
        # img_annotated = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result)
        # cv2.imwrite(os.path.join(save_path, 'temp_annotated.jpg'), img_annotated)

        # get ROI image
        img = cv2.imread(os.path.join(save_path, 'temp.jpg'), cv2.IMREAD_GRAYSCALE)
        img_ROI=get_ROI_image(img, detection_result)
        ed_time = time.time()
        print(f'ROI extractation time: {ed_time - st_time}s')
        cv2.imwrite(os.path.join(save_path, 'temp_ROI.jpg'), img_ROI)
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
        ed_time = time.time()
        print(f'Matching time: {ed_time - st_time}s')
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
        img = decode_base64_image(img_data)
        img.save(os.path.join(left_path, f'left_image_{i}.jpg'))

        # img = cv2.imread(os.path.join(left_path, f'left_image_{i}.jpg'), 1)
        # img = cv2.transpose(img)
        # img = cv2.flip(img, 1)
        # cv2.imwrite(os.path.join(left_path, f'left_image_{i}.jpg'), img)
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))
        detection_result = detector.detect(rgb_frame)

        img = cv2.imread(os.path.join(left_path, f'left_image_{i}.jpg'), cv2.IMREAD_GRAYSCALE)
        ROI_img = get_ROI_image(img, detection_result)
        cv2.imwrite(os.path.join(left_path, f'left_ROI_{i}.jpg'), ROI_img)

    right_path = os.path.join(save_path, 'right')
    if not os.path.exists(right_path):
        os.makedirs(right_path)
    for i, img_data in enumerate(right_images):
        img = decode_base64_image(img_data)
        img.save(os.path.join(right_path, f'right_image_{i}.jpg'))

        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))
        detection_result = detector.detect(rgb_frame)

        img = cv2.imread(os.path.join(right_path, f'right_image_{i}.jpg'), cv2.IMREAD_GRAYSCALE)
        ROI_img = get_ROI_image(img, detection_result)
        cv2.imwrite(os.path.join(right_path, f'right_ROI_{i}.jpg'), ROI_img)


    return jsonify({'code': 0})


@app.route('/recognize_native', methods=['POST'])
def reco():
    try:
        img_data = request.form.get('file')
    except Exception as e:
        return jsonify({'code': 1}), 401

    if img_data:
        st_time = time.time()
        save_path = os.path.join('./Users', app.config['TEST_SET'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # decode base64 image
        img = decode_base64_image(img_data)
        img.save(os.path.join(save_path, 'temp.png'))
        
        # get ROI image
        img_ROI= cv2.imread(os.path.join(save_path, 'temp.png'), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(save_path, 'temp_roi.jpg'), img_ROI)
        for root, dirs, files in os.walk('./Users'):
            if root == './Users':
                for dir in dirs:
                    if dir != 'test':
                        test_path = os.path.join('./Users', dir)
                        test_path = os.path.join(test_path, app.config['TRAIN_SET'] + '/left')
                        for i in range(2):
                            one_palmprint_code = encoder.encode_using_file(os.path.join(test_path, f'left_ROI_{i}.jpg'))
                            another_palmprint_code = encoder.encode_using_file(os.path.join(save_path, 'temp_roi.jpg'))
                            if one_palmprint_code.compare_to(another_palmprint_code) >= 0.12:
                                return jsonify({'code': 0, 'result': dir})

                        test_path = os.path.join('./Users', dir)
                        test_path = os.path.join(test_path, app.config['TRAIN_SET'] + '/right')
                        for i in range(2):
                            one_palmprint_code = encoder.encode_using_file(os.path.join(test_path, f'right_ROI_{i}.jpg'))
                            another_palmprint_code = encoder.encode_using_file(os.path.join(save_path, 'temp_roi.jpg'))
                            if one_palmprint_code.compare_to(another_palmprint_code) >= 0.12:
                                return jsonify({'code': 0, 'result': dir})

        ed_time = time.time()
        print(f'Matching time: {ed_time - st_time}s')

        return jsonify({'code': 0, 'result': 'None'})
    return jsonify({'code': 1}), 400


if __name__ == '__main__':
    context = ('cert.pem', 'key.pem')
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=context)
