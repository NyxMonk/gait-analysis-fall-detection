import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
from pathlib import Path

from src.config import BASE_DIR

init(autoreset=True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def section(title):
    print(Fore.CYAN + "\n" + "=" * 60)
    print(Fore.YELLOW + f" {title}")
    print(Fore.CYAN + "=" * 60 + "\n")

def success(msg):
    print(Fore.GREEN + f"✔ {msg}")

def warn(msg):
    print(Fore.YELLOW + f"⚠ {msg}")


def error(msg):
    print(Fore.RED + f"✖ {msg}")


def info(msg):
    print(Fore.CYAN + f"➤ {msg}")


# ---------- NATURAL SORTING ----------
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', s)]


# ---------- ANGLE CALCULATION ----------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def process_knee_video(video_path):
    info(f"Opening video: {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        error("Could not open video!")
        return None

    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    frame_angles = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)
            frame_angles.append(angle)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame, f"Angle: {angle:.2f}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)

        cv2.imshow("Knee Video Processing", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            warn("Skipped manually via Q key")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(frame_angles) == 0:
        warn("No valid pose frames detected.")
        return None

    success("Video processed successfully")

    return {
        "video": os.path.basename(video_path),
        "avg_angle": float(np.mean(frame_angles)),
        "max_angle": float(np.max(frame_angles)),
        "min_angle": float(np.min(frame_angles)),
        "frames_used": len(frame_angles)
    }

def analyze_knee_sessions(video_paths):
    section("Sorting Videos")
    video_paths = sorted(video_paths, key=natural_sort_key)
    for v in video_paths:
        info(f"→ {os.path.basename(v)}")

    section("Starting Session Analysis")

    session_data = []

    for vid in video_paths:
        print(Fore.MAGENTA + f"\n--- Processing {os.path.basename(vid)} ---")
        result = process_knee_video(vid)
        if result:
            result["session_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session_data.append(result)

    if not session_data:
        error("No valid videos processed!")
        return None

    df = pd.DataFrame(session_data)
    output_csv = BASE_DIR / "results" / "knee_sessions.csv"
    output_csv.parent.mkdir(exist_ok=True)

    df.to_csv(output_csv, index=False)
    success(f"CSV saved as {output_csv}")

    # ---------- IMPROVEMENT SUMMARY ----------
    if len(df) > 1:
        section("Improvement Summary")
        df["improvement"] = df["avg_angle"].diff().fillna(0)

        for i in range(1, len(df)):
            delta = df.loc[i, "improvement"]
            trend = (
                Fore.GREEN + "Improved" if delta > 2
                else Fore.RED + "Regressed" if delta < -2
                else Fore.YELLOW + "Stable"
            )
            print(f"Session {i}: Δ {delta:.2f}° → {trend}")

    # ---------- PROGRESS PLOT ----------
    section("Plotting Knee Progress Curve")
    plt.figure(figsize=(7, 4))
    plt.plot(df["avg_angle"], marker='o')
    plt.title("Knee Exercise Progress")
    plt.xlabel("Session Number")
    plt.ylabel("Average Knee Angle (°)")
    plt.grid(True)
    plt.show()

    return df


# ---------- MAIN EXECUTION ----------
section("Knee Angle Analysis Script")

BASE_DIR = Path(__file__).resolve().parent.parent
video_folder = BASE_DIR / "data" / "videos"

if not video_folder.exists():
    error("Video folder not found: data/videos")
    raise SystemExit

video_paths = list(video_folder.glob("*.mp4")) + \
              list(video_folder.glob("*.avi")) + \
              list(video_folder.glob("*.mov")) + \
              list(video_folder.glob("*.mkv"))

if not video_paths:
    error("No video files found in the folder!")
else:
    analyze_knee_sessions(video_paths)
