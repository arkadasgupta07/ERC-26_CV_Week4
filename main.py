import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import pygame

pygame.mixer.init()
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky finger
    (0, 17)                                 # Palm base connection
]

sounds_mapping = {
    1: r"sounds\cr78-Bongo High.mp3",
    2: r"sounds\cr78-Cymbal.mp3",
    3: r"sounds\cr78-Guiro 1.mp3",
    4: r"sounds\crash.mp3",
    5: r"sounds\kick-bass.mp3",
    6: r"sounds\snare.mp3",
    7: r"sounds\tempest-HiHat Metal.mp3",
    8: r"sounds\tom-1.mp3",
    9: r"sounds\tom-2.mp3",
    10: r"sounds\tom-3.mp3"
}

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    # Convert each landmark to pixel coordinates
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    # Draw lines between connected landmarks (green)
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0,255,0), 2)
    # Draw a dot at each landmark (red, filled)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0,0,255), -1)

def count_fingers(hand_landmarks, handedness_label):
    count = 0
    # indices of the tips and pips of four fingers
    tips_id = [8, 12, 16, 20]
    pip_id = [6, 10, 14, 18]    

    # counting raised fingers of the four fingers
    for tip, pip in zip(tips_id, pip_id):
        if hand_landmarks[tip].y < hand_landmarks[pip].y: # if coordinate is lower then its above
            count += 1

    # indices for thumb
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]

    if handedness_label == 'Left':
        if thumb_tip.x < thumb_ip.x: # tip to the left of the ip
            count += 1

    if handedness_label == 'Right':
        if thumb_tip.x > thumb_ip.x:
            count += 1

    return count

base_options = mp_python.BaseOptions(model_asset_path = "hand_landmarker.task")

options = mp_vision.HandLandmarkerOptions(
        base_options = base_options,
        running_mode = mp_vision.RunningMode.VIDEO,
        num_hands = 2
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
start_time = time.time()

is_muted = False

while cap.isOpened():

    count = 0

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # 1 flips horizontally, 0 vertically and -1 both simultaneously

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame_rgb)

    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks and result.handedness: # 0 if no hands detected

        for i, hand_landmarks in enumerate(result.hand_landmarks):

            draw_landmarks(frame, hand_landmarks)
            label = result.handedness[i][0].category_name

            count += count_fingers(hand_landmarks, label)

    if count in sounds_mapping:

        pygame.mixer.music.load(sounds_mapping[count])
        pygame.mixer.music.play()

        cv2.putText(frame,
                    f"Sound: {sounds_mapping[count][8:-4]}",
                    (50,80),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255,0,0),
                    2
        )

    cv2.putText(frame,
                f"Fingers: {count}",
                (50,450),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0,0,0),
                2
    )

    cv2.putText(frame,
                f"Volume: {0 if is_muted else 100}",
                (450, 450),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0,255,200),
                2
    )

    cv2.imshow("Gesture Media Player", frame)
    
    key = cv2.waitKey(1) & 0xFF # ord converts the char into ascii
    if key == ord('m') or key == ord('M'):
        is_muted = not is_muted
        if is_muted:
            pygame.mixer.music.set_volume(0.0)
        else: 
            pygame.mixer.music.set_volume(1.0)

    if key == ord('q') or key == ord('Q'):
            break

cap.release()
cv2.destroyAllWindows()
landmarker.close()