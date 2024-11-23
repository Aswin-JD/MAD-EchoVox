import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
from gtts import gTTS
import pygame
from PIL import Image
import tempfile
import time
from gtts import gTTS
import pygame

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Streamlit app configuration
st.title("Sign Language Recognition")
st.sidebar.header("Settings")
st.sidebar.text("Configure the application here.")

# Load the model and other required data
pq_file = '100015657.parquet'
pq = pd.read_parquet(pq_file)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
prediction_fn = interpreter.get_signature_runner("serving_default")

train = pd.read_csv('train.csv')
train['sign_ord'] = train['sign'].astype('category').cat.codes
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

# Define helper functions
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

    landmarks = pd.concat([
        face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face'),
        pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose'),
        left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand'),
        right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')
    ]).reset_index(drop=True)

    landmarks = pq_skel.merge(landmarks, on=['type', 'landmark_index'], how='left').assign(frame=frame)
    return landmarks

def get_prediction(prediction_fn, landmarks):
    ROWS_PER_FRAME = 543
    data = landmarks.values
    n_frames = len(data) // ROWS_PER_FRAME
    data = data.reshape(n_frames, ROWS_PER_FRAME, 3).astype(np.float32)
    prediction = prediction_fn(inputs=data)
    pred = prediction['outputs'].argmax()
    pred_conf = prediction['outputs'][pred]

    if np.isnan(pred_conf) or pred_conf < 0.075:
        st.warning("Prediction confidence is too low. Ignoring...")
        return None, None

    sign = ORD2SIGN[pred]
    return sign, pred_conf

# Streamlit video capture and processing
st.header("Live Camera Feed")

run = st.button("Start", key='start_button')
if run:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    all_landmarks = []
    recognized_text = st.empty()  # To display the recognized sign

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            current_time = time.time()
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera not detected. Please check your device.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            landmarks = create_frame_landmark_df(results, 0, pq)
            all_landmarks.append(landmarks)

            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Display the frame in Streamlit
            stframe.image(image, channels="RGB")

            if current_time - start_time >= 5:
                if all_landmarks:
                    landmarks_df = pd.concat(all_landmarks).reset_index(drop=True)
                    new_landmarks = landmarks_df.drop(['type', 'landmark_index', 'frame'], axis=1)
                    output, confidence = get_prediction(prediction_fn, new_landmarks)
                    if output:
                        recognized_text.text(f"Recognized Sign: {output} with Confidence: {confidence:.2f}")
                        tts = gTTS(text=output, lang='en')
                        tts.save("output.mp3")
                        pygame.mixer.init()
                        pygame.mixer.music.load("output.mp3")
                        pygame.mixer.music.play()
                    else:
                        recognized_text.text("No valid sign detected.")

                all_landmarks = []
                start_time = current_time  # Reset timer
    cap.release()
