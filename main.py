import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pygame

pygame.mixer.init()
MODEL_PATH = 'hand_landmarker.task'
SWARA_FILES = {
    1: 'Sa.mp3',
    2: 'Re.mp3',
    3: 'Ga.mp3',
    4: 'Ma.mp3',
    5: 'Pa.mp3',
    6: 'Dha.mp3',
    7: 'Ni.mp3',
    8: 'Sa_upper.mp3',
    10: 'Fa.mp3'
}

sounds = {}
for count, filename in SWARA_FILES.items():
    sound_path = "sounds_swaras/" + filename
    try:
        sounds[count] = pygame.mixer.Sound(sound_path)
    except Exception as e:
        print(f"[WARNING] Could not load sound file: {sound_path}. Error: {e}")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         #Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         #Index
    (5, 9), (9, 10), (10, 11), (11, 12),    #Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  #Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  #Pinky & Palm base
]

def count_all_raised_fingers(multi_hand_landmarks):
    total_fingers = 0
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    
    for hand_landmarks in multi_hand_landmarks:
        for tip, pip in zip(tip_ids, pip_ids):
            if hand_landmarks[tip].y < hand_landmarks[pip].y:
                total_fingers += 1                
        if abs(hand_landmarks[4].x - hand_landmarks[17].x) > abs(hand_landmarks[3].x - hand_landmarks[17].x):
            total_fingers += 1
            
    return total_fingers

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = []
    
    for lm in hand_landmarks:
        lx = lm.x if hasattr(lm, 'x') else lm[0]
        ly = lm.y if hasattr(lm, 'y') else lm[1]
        points.append((int(lx * w), int(ly * h)))
    
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
            
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

try:
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
except Exception as e:
    print(f"\n[ERROR] Could not load model from '{MODEL_PATH}'.")
    exit(1)

cap = cv2.VideoCapture(0)
last_timestamp_ms = -1
last_played_finger = -1

print("[INFO] Starting Gesture Music Player. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    timestamp_ms = int(time.monotonic() * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    current_fingers = 0
    if result and result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)
        current_fingers = count_all_raised_fingers(result.hand_landmarks)

    if current_fingers in [0, 9]:
        if last_played_finger != 0:
            pygame.mixer.stop()
            last_played_finger = 0
    elif current_fingers in sounds:
        if current_fingers != last_played_finger:
            pygame.mixer.stop() 
            sounds[current_fingers].play()
            last_played_finger = current_fingers

    status_text = f"Fingers: {current_fingers} | Swara: {SWARA_FILES.get(current_fingers, 'Silence')}"
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('Gesture Music Player', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()