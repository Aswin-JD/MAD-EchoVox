import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
from gtts import gTTS
import pygame
import warnings
import time
import firebase_admin
from firebase_admin import credentials, db

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://echovox-f0a6b-default-rtdb.firebaseio.com/'  # Replace with your database URL
})

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Function to create landmark dataframe
def create_frame_landmark_df(results, frame, pq):
    pq_skel = pq[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    face, pose, left_hand, right_hand = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    # Combine dataframes and reset index
    landmarks = pd.concat([
        face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face'),
        pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose'),
        left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand'),
        right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')
    ]).reset_index(drop=True)

    # Merge with skeleton and add frame info
    landmarks = pq_skel.merge(landmarks, on=['type', 'landmark_index'], how='left').assign(frame=frame)
    return landmarks

# Function to capture and process video for sign recognition
def do_capture_loop(pq):
    all_landmarks = []
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    # Initialize Mediapipe holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame = 0

        while cap.isOpened():
            current_time = time.time()
            frame += 1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Preprocess image for holistic processing
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Create landmark dataframe
            landmarks = create_frame_landmark_df(results, frame, pq)
            all_landmarks.append(landmarks)

            # Display video with landmarks
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('Capture Sign', cv2.flip(image, 1))

            # Process landmarks every 5 seconds
            if current_time - start_time >= 5:
                if all_landmarks:  # Ensure there are landmarks to process
                    landmarks_df = pd.concat(all_landmarks).reset_index(drop=True)
                    new_landmarks = landmarks_df.drop(['type', 'landmark_index', 'frame'], axis=1)
                    output, confidence = get_prediction(prediction_fn, new_landmarks)

                    # Text-to-Speech for recognized sign
                    if output is not None:
                        tts = gTTS(text=output, lang='en')
                        tts.save("output.mp3")
                        pygame.mixer.init()
                        pygame.mixer.music.load("output.mp3")
                        pygame.mixer.music.play()

                        # Update Firebase
                        update_firebase(output, confidence)

                # Reset for next interval
                all_landmarks = []
                start_time = current_time  # Reset timer

            # Exit on 'Esc' key
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# TensorFlow Lite Model Prediction Function
def get_prediction(prediction_fn, landmarks):
    ROWS_PER_FRAME = 543
    data = landmarks.values
    n_frames = len(data) // ROWS_PER_FRAME
    data = data.reshape(n_frames, ROWS_PER_FRAME, 3).astype(np.float32)
    prediction = prediction_fn(inputs=data)
    pred = prediction['outputs'].argmax()
    pred_conf = prediction['outputs'][pred]

    # Ignore predictions with confidence NaN or below 0.10
    if np.isnan(pred_conf) or pred_conf < 0.075:
        print("Prediction confidence is too low. Ignoring...")
        return None, None

    sign = ORD2SIGN[pred]
    print(f'Sign: {sign} with confidence {pred_conf:.4f}')
    return sign, pred_conf

# Function to update Firebase
def update_firebase(output, confidence):
    
    # Convert numpy.float32 to Python float
    confidence = float(confidence)  # Ensure confidence is a Python float
    
    ref = db.reference('sign_predictions')  # Replace with your actual Firebase path
    ref.set({
        'sign': output,
        'confidence': confidence,
        'timestamp': time.time()
    })
    print(f"Firebase updated with: Sign = {output}, Confidence = {confidence:.4f}")


# Load required data
pq_file = '100015657.parquet'
pq = pd.read_parquet(pq_file)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
prediction_fn = interpreter.get_signature_runner("serving_default")

train = pd.read_csv('train.csv')
train['sign_ord'] = train['sign'].astype('category').cat.codes
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

# Run the sign recognition loop
do_capture_loop(pq)