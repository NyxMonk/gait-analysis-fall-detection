from pathlib import Path
import cv2
import numpy as np
import math
import os
import csv
from colorama import init, Fore, Style
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

init(autoreset=True)

# Pose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Create pose landmarker - with fallback if model is not available
def create_pose_landmarker():
    try:
        model_path = 'pose_landmarker_full.tflite'
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_confidence=True,
            num_poses=1
        )
        return vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Warning: Could not load pose landmarker model: {e}")
        return None

pose_landmarker = create_pose_landmarker()

def get_landmarks(frame):
    """Extract pose landmarks from frame"""
    if pose_landmarker is None:
        return None
    
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = pose_landmarker.detect(mp_image)
        
        if detection_result.pose_landmarks:
            return detection_result.pose_landmarks[0]
    except Exception as e:
        print(f"Error processing frame: {e}")
    
    return None

def classify_video(video_path, streak_threshold=3):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return "UNKNOWN", "Unknown"

    window_name = f"Processing {os.path.basename(video_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    prev_left_ankle_x, prev_right_ankle_x = None, None
    fall_streak = 0
    fall_confirmed = False
    video_status = "NOT FALL"
    impact_area = "Unknown"
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process every 5th frame for performance
        if frame_count % 5 != 0:
            continue

        landmarks = get_landmarks(frame)
        status = "Unknown"

        if landmarks:
            # Get landmark positions
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            left_hip = landmarks[LEFT_HIP]
            right_hip = landmarks[RIGHT_HIP]

            mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            mid_hip_x = (left_hip.x + right_hip.x) / 2
            mid_hip_y = (left_hip.y + right_hip.y) / 2

            dx = mid_shoulder_x - mid_hip_x
            dy = mid_shoulder_y - mid_hip_y
            angle = abs(math.degrees(math.atan2(dy, dx)))

            left_ankle = landmarks[LEFT_ANKLE]
            right_ankle = landmarks[RIGHT_ANKLE]

            if angle < 45 or angle > 135:
                fall_streak += 1
                if fall_streak >= streak_threshold:
                    fall_confirmed = True
                    status = "FALL"
                    video_status = "FALL"

                    key_landmarks = {
                        "HEAD": landmarks[NOSE].y,
                        "LEFT_HAND": landmarks[LEFT_WRIST].y,
                        "RIGHT_HAND": landmarks[RIGHT_WRIST].y,
                        "LEFT_KNEE": landmarks[LEFT_KNEE].y,
                        "RIGHT_KNEE": landmarks[RIGHT_KNEE].y,
                        "LEFT_FOOT": landmarks[LEFT_FOOT_INDEX].y,
                        "RIGHT_FOOT": landmarks[RIGHT_FOOT_INDEX].y,
                    }
                    impact_area = max(key_landmarks, key=key_landmarks.get)

            else:
                fall_streak = 0
                if prev_left_ankle_x and prev_right_ankle_x:
                    move_left = abs(left_ankle.x - prev_left_ankle_x)
                    move_right = abs(right_ankle.x - prev_right_ankle_x)

                    if move_left > 0.02 or move_right > 0.02:
                        status = "WALKING"
                        if not fall_confirmed:
                            video_status = "NOT FALL"
                    else:
                        status = "STANDING"
                        if not fall_confirmed:
                            video_status = "NOT FALL"
                else:
                    status = "STANDING"

            prev_left_ankle_x, prev_right_ankle_x = left_ankle.x, right_ankle.x

        cv2.putText(frame, status, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 255) if status == "FALL" else (0, 255, 0), 3)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if video_status == "FALL":
        status_console = f"{Fore.RED}FALL{Style.RESET_ALL}"
    elif status == "WALKING":
        status_console = f"{Fore.YELLOW}WALKING{Style.RESET_ALL}"
    else:
        status_console = f"{Fore.GREEN}STANDING / NOT FALL{Style.RESET_ALL}"

    print(f"\n{'='*60}")
    print(f"{Fore.CYAN}Video: {os.path.basename(video_path)}{Style.RESET_ALL}")
    print(f"Status      : {status_console}")
    print(f"Impact Area : {Fore.MAGENTA}{impact_area}{Style.RESET_ALL}")
    print(f"{'='*60}\n")

    return video_status, impact_area


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    folder_path = BASE_DIR / "data" / "fall_videos"
    output_csv = BASE_DIR / "results" / "fall_detection_results.csv"
    output_csv.parent.mkdir(exist_ok=True)

    # Check if folder exists and has videos
    if not folder_path.exists():
        print(f"Folder {folder_path} does not exist. Creating it...")
        folder_path.mkdir(parents=True, exist_ok=True)
    
    video_files = list(folder_path.glob("*.mp4"))
    if not video_files:
        print(f"No MP4 videos found in {folder_path}")
    else:
        print(f"Found {len(video_files)} video(s) to process")

        with open(output_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Video Name", "Fall Status", "Impact Area"])

            for video_path in video_files:
                status, impact = classify_video(str(video_path))
                writer.writerow([video_path.name, status, impact])

        print(f"{Fore.GREEN}Results saved to {output_csv}{Style.RESET_ALL}")
