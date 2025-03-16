import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
import math
import time
import os
  
# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Webcam input
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Video recording setup
video_writer = None
recording = False

# Snapshot directory setup
snapshot_dir = "C:\\Users\\viswa\\OneDrive\\Documents\\hand_gesters"
os.makedirs(snapshot_dir, exist_ok=True)

# Variables for gesture tracking
zoom_level = 1.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(rgb_frame)
    zoomed_frame = frame  # Default zoomed frame in case no hand is detected

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for all fingers
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Convert normalized coordinates to pixel values
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])

            # Volume control (thumb up/down gesture)
            thumb_up = (thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and
                        thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y)
            thumb_down = (thumb_tip.y > index_tip.y and thumb_tip.y > middle_tip.y and
                          thumb_tip.y > ring_tip.y and thumb_tip.y > pinky_tip.y)

            if thumb_up:
                current_volume = volume.GetMasterVolumeLevelScalar()
                volume_level = min(current_volume + 0.1, 1.0)  # Increase volume by 10%
                volume.SetMasterVolumeLevelScalar(volume_level, None)
                cv2.putText(frame, "Volume Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif thumb_down:
                current_volume = volume.GetMasterVolumeLevelScalar()
                volume_level = max(current_volume - 0.1, 0.0)  # Decrease volume by 10%
                volume.SetMasterVolumeLevelScalar(volume_level, None)
                cv2.putText(frame, "Volume Down", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Picture taking (index and middle fingers extended)
            index_and_middle = (index_tip.y < middle_tip.y and index_tip.y < ring_tip.y and
                                index_tip.y < pinky_tip.y and middle_tip.y < ring_tip.y and
                                middle_tip.y < pinky_tip.y)

            if index_and_middle:
                # Take a snapshot
                cv2.putText(frame, "Picture Taken!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                snapshot_path = os.path.join(snapshot_dir, f"Snapshot_{int(time.time())}.png")
                cv2.imwrite(snapshot_path, frame)
                time.sleep(0.5)  # Prevent multiple snapshots for a single gesture

            # Start/Stop video recording (middle, index, and pinky extended)
            start_stop_video = (index_tip.y < middle_tip.y and middle_tip.y < ring_tip.y and pinky_tip.y < ring_tip.y)

            if start_stop_video:
                if not recording:
                    video_path = f"Video_{int(time.time())}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    recording = True
                    cv2.putText(frame, "Recording Started", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    recording = False
                    video_writer.release()
                    cv2.putText(frame, "Recording Stopped", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Zoom Control (index and thumb distance)
            thumb_index_distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            # Set zoom level based on thumb-index distance
            if thumb_index_distance < 150:  # Zoom Out gesture (spread fingers)
                zoom_level = min(5.0, zoom_level + 0.02)  # Zoom out by increasing zoom level (more precise)
                cv2.putText(frame, "Zoom Out", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif thumb_index_distance > 50:  # Zoom In gesture (close fingers)
                zoom_level = max(0.5, zoom_level - 0.02)  # Zoom in by decreasing zoom level (more precise)
                cv2.putText(frame, "Zoom In", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Apply zoom to the frame
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            zoomed_width = int(width * zoom_level)
            zoomed_height = int(height * zoom_level)
            zoomed_frame = frame[
                max(center_y - zoomed_height // 2, 0):min(center_y + zoomed_height // 2, height),
                max(center_x - zoomed_width // 2, 0):min(center_x + zoomed_width // 2, width)
            ]
            zoomed_frame = cv2.resize(zoomed_frame, (width, height))  # Resize back to original frame size

            # Write video if recording
            if recording and video_writer is not None:
                video_writer.write(frame)

    # Display the frame
    cv2.imshow("Gesture Control", zoomed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
if recording and video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
