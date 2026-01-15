from __future__ import annotations
import os
from zipfile import Path
import cv2
import glob
import json
import math
import argparse
from collections import deque, defaultdict
from typing import Any, Dict, List
from pathlib import Path


import numpy as np
import pandas as pd

from colorama import Fore, Style, init

from src.config import BASE_DIR
init(autoreset=True)

def pretty_print_result(video_name, feats):
    print(Fore.CYAN + "────────────────────────────────────────────")
    print(Fore.YELLOW + f"VIDEO : {video_name}")
    print(Fore.GREEN  + f"LABEL : {feats['rule_label']}")
    print(Fore.MAGENTA + f"RULE HITS : {feats['rule_hits']}")
    print(Fore.BLUE + "THERAPY SUMMARY:")
    print(Fore.WHITE + feats['plan_summary'])
    print(Fore.CYAN + "────────────────────────────────────────────\n")

# ---- Quieter logs (set BEFORE importing mediapipe) ----
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def np_point(lm):
    return np.array([lm.x, lm.y])


def angle_deg(a, b, c):
    a = np_point(a)
    b = np_point(b)
    c = np_point(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def dist(a, b):
    a = np_point(a)
    b = np_point(b)
    return float(np.linalg.norm(a - b))


def smooth_series(x, win=5):
    if len(x) < 3 or win <= 1:
        return np.asarray(x, dtype=float)
    win = min(win, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if win < 3:
        return np.asarray(x, dtype=float)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(win) / win
    return np.convolve(xpad, kernel, mode='valid')


def detect_peaks_max(y, min_distance_frames=8):
    """Detect local maxima indices (ankle y is larger at contact since y grows downward)."""
    y = np.asarray(y)
    n = len(y)
    peaks = []
    last_idx = -10**9
    for i in range(1, n - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            if i - last_idx >= min_distance_frames:
                peaks.append(i)
                last_idx = i
    return peaks


def symmetry_index(l, r):
    denom = 0.5 * (l + r) + 1e-8
    return abs(l - r) / denom


class GaitFeatureExtractor:
    def __init__(self, min_det_conf=0.5, min_track_conf=0.5, draw=False, show=False):
        self.pose = mp_pose.Pose(min_detection_confidence=min_det_conf,
                                 min_tracking_confidence=min_track_conf)
        self.draw = draw
        self.show = show

    def _scale_reference(self, lm):
        """Return a person scale (approximate body size) for normalization."""
        l_ank, r_ank = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value], lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        l_knee, r_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value], lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        l_hip, r_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value], lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        l_sho, r_sho = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        leg_l = dist(l_hip, l_ank)
        leg_r = dist(r_hip, r_ank)
        trunk_l = dist(l_sho, l_hip)
        trunk_r = dist(r_sho, r_hip)
        return max(1e-6, np.mean([leg_l, leg_r, trunk_l, trunk_r]))

    def process_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 30.0  # fallback

        L_ank_x, L_ank_y = [], []
        R_ank_x, R_ank_y = [], []
        L_knee_ang, R_knee_ang = [], []
        L_hip_ang, R_hip_ang = [], []
        L_ank_ang, R_ank_ang = [], []
        scales = []
        step_widths = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                l_ank = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                r_ank = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
                r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
                l_toe = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
                r_toe = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

                # Append series
                L_ank_x.append(l_ank.x); L_ank_y.append(l_ank.y)
                R_ank_x.append(r_ank.x); R_ank_y.append(r_ank.y)

                # Joint angles
                L_knee_ang.append(angle_deg(l_hip, l_knee, l_ank))
                R_knee_ang.append(angle_deg(r_hip, r_knee, r_ank))
                l_sho = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                r_sho = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                L_hip_ang.append(angle_deg(l_sho, l_hip, l_knee))
                R_hip_ang.append(angle_deg(r_sho, r_hip, r_knee))
                L_ank_ang.append(angle_deg(l_knee, l_ank, l_toe))
                R_ank_ang.append(angle_deg(r_knee, r_ank, r_toe))

                # Person scale (for normalization)
                scales.append(self._scale_reference(lm))

                # Step width ~ horizontal distance between ankles (x)
                step_widths.append(abs(l_ank.x - r_ank.x))

                if self.draw or self.show:
                    mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if self.show:
                cv2.imshow('Gait (MediaPipe)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if self.show:
            cv2.destroyAllWindows()

        # Convert to numpy
        L_ank_x = np.asarray(L_ank_x); L_ank_y = np.asarray(L_ank_y)
        R_ank_x = np.asarray(R_ank_x); R_ank_y = np.asarray(R_ank_y)
        L_knee_ang = np.asarray(L_knee_ang); R_knee_ang = np.asarray(R_knee_ang)
        L_hip_ang = np.asarray(L_hip_ang); R_hip_ang = np.asarray(R_hip_ang)
        L_ank_ang = np.asarray(L_ank_ang); R_ank_ang = np.asarray(R_ank_ang)
        scales = np.asarray(scales) if len(scales) else np.array([1.0])
        step_widths = np.asarray(step_widths)


        L_y_sm = smooth_series(L_ank_y, win=7)
        R_y_sm = smooth_series(R_ank_y, win=7)


        min_step_frames = int(max(6, fps * 0.25))  # at least 0.25s between strikes
        L_strikes = detect_peaks_max(L_y_sm, min_distance_frames=min_step_frames)
        R_strikes = detect_peaks_max(R_y_sm, min_distance_frames=min_step_frames)

        # Compute features
        feats = {}
        feats.update(self._spatiotemporal(L_ank_x, R_ank_x, L_strikes, R_strikes, fps, scales, step_widths))
        feats.update(self._kinematic(L_knee_ang, R_knee_ang, L_hip_ang, R_hip_ang, L_ank_ang, R_ank_ang, L_strikes, R_strikes, fps))
        feats.update(self._kinetic_proxies(L_y_sm, R_y_sm, fps))
        feats.update(self._symmetry(feats))

        # Low-detection guard (skip super-poor detections)
        min_frames_needed = 30  # ~1 second at 30 fps
        if len(scales) < min_frames_needed:
            raise RuntimeError(f"Too few pose frames detected ({len(scales)}). Skipping.")

        # Rule-based classification (heuristic)
        label, rule_hits = classify_rules(feats)
        feats['rule_label'] = label
        feats['rule_hits'] = ', '.join(rule_hits)

        # ----- Therapy recommendation -----
        plan = recommend_from_feats(feats)
        feats['plan_summary'] = plan.get('summary', '')
        feats['plan_markdown'] = plan.get('plan_markdown', '')
        # -----------------------------------------

        return feats

    # -------------------------
    # Spatiotemporal features
    # -------------------------
    def _spatiotemporal(self, Lx, Rx, L_strikes, R_strikes, fps, scales, step_widths):
        T = len(Lx)
        time = np.arange(T) / fps if T else np.array([])
        scale_med = float(np.median(scales)) if len(scales) else 1.0

        # Cadence (steps/min)
        total_steps = len(L_strikes) + len(R_strikes)
        duration_sec = float(time[-1] - time[0]) if T > 1 else 0.0
        cadence = (total_steps / max(1e-6, duration_sec)) * 60.0 if duration_sec > 0 else np.nan

        # Helpers
        def diffs_in_time(idxs):
            idxs = np.asarray(idxs, dtype=int)
            return (np.diff(idxs) / fps) if idxs.size >= 2 else np.array([])

        L_stride_times = diffs_in_time(L_strikes)
        R_stride_times = diffs_in_time(R_strikes)

        # Step time L→R and R→L
        step_times = []
        i = j = 0
        Ls = np.asarray(L_strikes, dtype=int)
        Rs = np.asarray(R_strikes, dtype=int)
        while i < len(Ls) and j < len(Rs):
            if Ls[i] < Rs[j]:
                step_times.append((Rs[j] - Ls[i]) / fps); i += 1
            else:
                step_times.append((Ls[i] - Rs[j]) / fps); j += 1
        step_times = np.asarray(step_times)

        # Stride length proxy from ankle-x displacement
        def stride_len(x, idxs):
            idxs = list(map(int, idxs))
            vals = []
            for k in range(1, len(idxs)):
                vals.append(abs(x[idxs[k]] - x[idxs[k-1]]))
            return np.asarray(vals)

        L_stride_len = stride_len(Lx, L_strikes)
        R_stride_len = stride_len(Rx, R_strikes)

        denom = max(1e-6, scale_med)
        L_stride_len_n = L_stride_len / denom if L_stride_len.size else np.array([])
        R_stride_len_n = R_stride_len / denom if R_stride_len.size else np.array([])
        step_widths_n  = step_widths  / denom if len(step_widths) else np.array([])

        feats = {
            'fps': float(fps),
            'duration_s': float(duration_sec),
            'cadence_spm': float(cadence) if not np.isnan(cadence) else np.nan,
            'L_stride_time_mean_s': float(np.mean(L_stride_times)) if L_stride_times.size else np.nan,
            'R_stride_time_mean_s': float(np.mean(R_stride_times)) if R_stride_times.size else np.nan,
            'step_time_mean_s': float(np.mean(step_times)) if step_times.size else np.nan,
            'step_time_cv': float(np.std(step_times) / (np.mean(step_times) + 1e-8)) if step_times.size else np.nan,
            'L_stride_len_norm_mean': float(np.mean(L_stride_len_n)) if L_stride_len_n.size else np.nan,
            'R_stride_len_norm_mean': float(np.mean(R_stride_len_n)) if R_stride_len_n.size else np.nan,
            'step_width_norm_mean': float(np.mean(step_widths_n)) if step_widths_n.size else np.nan,
            'step_width_norm_std': float(np.std(step_widths_n)) if step_widths_n.size else np.nan,
            '_L_strikes_idx': json.dumps(list(map(int, L_strikes))),
            '_R_strikes_idx': json.dumps(list(map(int, R_strikes))),
        }
        return feats

    # -------------------------
    # Kinematic features (angles)
    # -------------------------
    def _kinematic(self, L_knee, R_knee, L_hip, R_hip, L_ank, R_ank, L_strikes, R_strikes, fps):
        def rom_stats(sig):
            if len(sig) == 0:
                return (np.nan, np.nan, np.nan)
            return float(np.min(sig)), float(np.max(sig)), float(np.max(sig) - np.min(sig))

        feats = {}
        for name, sig in [('L_knee', L_knee), ('R_knee', R_knee), ('L_hip', L_hip), ('R_hip', R_hip), ('L_ank', L_ank), ('R_ank', R_ank)]:
            mn, mx, rom = rom_stats(sig)
            feats[f'{name}_angle_min_deg'] = mn
            feats[f'{name}_angle_max_deg'] = mx
            feats[f'{name}_ROM_deg'] = rom
            feats[f'{name}_mean_deg'] = float(np.mean(sig)) if len(sig) else np.nan
            feats[f'{name}_std_deg'] = float(np.std(sig)) if len(sig) else np.nan
        return feats

    # -------------------------
    # Kinetic proxies (from acceleration)
    # -------------------------
    def _kinetic_proxies(self, L_y_sm, R_y_sm, fps):
        # Vertical velocity & acceleration (normalized units per second)
        Ly = np.asarray(L_y_sm, dtype=float)
        Ry = np.asarray(R_y_sm, dtype=float)
        if Ly.size < 3 or Ry.size < 3:
            return {
                'L_vert_accel_peak': np.nan,
                'R_vert_accel_peak': np.nan,
                'L_vert_accel_rms': np.nan,
                'R_vert_accel_rms': np.nan,
            }
        Lv = np.gradient(Ly) * fps
        Rv = np.gradient(Ry) * fps
        La = np.gradient(Lv) * fps
        Ra = np.gradient(Rv) * fps
        return {
            'L_vert_accel_peak': float(np.max(np.abs(La))),
            'R_vert_accel_peak': float(np.max(np.abs(Ra))),
            'L_vert_accel_rms': float(np.sqrt(np.mean(La**2))),
            'R_vert_accel_rms': float(np.sqrt(np.mean(Ra**2))),
        }

    # -------------------------
    # Symmetry features (computed from existing metrics)
    # -------------------------
    def _symmetry(self, feats):
        L_t = feats.get('L_stride_time_mean_s', np.nan)
        R_t = feats.get('R_stride_time_mean_s', np.nan)
        L_l = feats.get('L_stride_len_norm_mean', np.nan)
        R_l = feats.get('R_stride_len_norm_mean', np.nan)
        si_time = symmetry_index(L_t, R_t) if (not np.isnan(L_t) and not np.isnan(R_t)) else np.nan
        si_len  = symmetry_index(L_l, R_l) if (not np.isnan(L_l) and not np.isnan(R_l)) else np.nan
        feats.update({
            'symmetry_stride_time_SI': float(si_time) if not np.isnan(si_time) else np.nan,
            'symmetry_stride_len_SI': float(si_len) if not np.isnan(si_len) else np.nan,
        })
        return feats


# -----------------------------
# Rule-based classifier (heuristic)
# -----------------------------

def classify_rules(f):
    hits = []
    scores = {
        "parkinsonian": 0,
        "hemiplegic": 0,
        "ataxic": 0,
        "spastic": 0,
        "antalgic": 0
    }

    def g(k, d=np.nan):
        return f.get(k, d)

    # --- extracted features ---
    cadence = g('cadence_spm')
    step_cv = g('step_time_cv')
    stepw = g('step_width_norm_mean')
    L_rom_knee = g('L_knee_ROM_deg'); R_rom_knee = g('R_knee_ROM_deg')
    L_rom_ank = g('L_ank_ROM_deg');  R_rom_ank = g('R_ank_ROM_deg')
    si_time = g('symmetry_stride_time_SI')
    si_len  = g('symmetry_stride_len_SI')
    L_acc_pk = g('L_vert_accel_peak'); R_acc_pk = g('R_vert_accel_peak')
    mean_stride_len = np.nanmean([g('L_stride_len_norm_mean'), g('R_stride_len_norm_mean')])
    min_knee_rom = np.nanmin([L_rom_knee, R_rom_knee])
    min_ank_rom  = np.nanmin([L_rom_ank, R_rom_ank])

    # --- THRESHOLDS (more forgiving) ---
    SHORT_STRIDE = 0.15
    LOW_KNEE_ROM = 30
    LOW_ANK_ROM = 20
    HIGH_STEP_VAR = 0.30
    ASYMM = 0.30
    WIDE_STEP = 0.28
    LOW_CADENCE = 80

    # -----------------------------
    # Parkinsonian-like
    # -----------------------------
    if not np.isnan(mean_stride_len) and mean_stride_len < SHORT_STRIDE:
        scores["parkinsonian"] += 2
    if not np.isnan(min_knee_rom) and min_knee_rom < LOW_KNEE_ROM:
        scores["parkinsonian"] += 1
    if not np.isnan(cadence) and cadence < LOW_CADENCE:
        scores["parkinsonian"] += 1

    # -----------------------------
    # Hemiplegic-like
    # -----------------------------
    if not np.isnan(si_time) and si_time > ASYMM:
        scores["hemiplegic"] += 2
    if not np.isnan(si_len) and si_len > ASYMM:
        scores["hemiplegic"] += 1

    # -----------------------------
    # Ataxic-like
    # -----------------------------
    if not np.isnan(stepw) and stepw > WIDE_STEP:
        scores["ataxic"] += 2
    if not np.isnan(step_cv) and step_cv > HIGH_STEP_VAR:
        scores["ataxic"] += 1

    # -----------------------------
    # Spastic-like
    # -----------------------------
    if not np.isnan(min_ank_rom) and min_ank_rom < LOW_ANK_ROM:
        scores["spastic"] += 2
    if not np.isnan(min_knee_rom) and min_knee_rom < LOW_KNEE_ROM:
        scores["spastic"] += 1

    # -----------------------------
    # Antalgic-like
    # -----------------------------
    if not (np.isnan(L_acc_pk) or np.isnan(R_acc_pk)):
        if abs(L_acc_pk - R_acc_pk) > 2 * np.nanstd([L_acc_pk, R_acc_pk]):
            scores["antalgic"] += 2

    # -----------------------------
    # DECISION
    # -----------------------------
    final = []
    for k, v in scores.items():
        if v >= 3:  # require strong evidence
            final.append(k + "-like")

    if not final:
        return ("normal/unclear", [])

    return ("; ".join(final), final)



# -----------------------------
# Therapy Knowledge Base & API
# -----------------------------

THERAPY_DB: Dict[str, Dict[str, Any]] = {
    "parkinsonian": {
        "goals": [
            "Increase stride length and arm swing",
            "Improve rhythm and initiation (reduce freezing)",
            "Upright posture and dynamic balance",
        ],
        "precautions": [
            "Watch for freezing at doorways/turns; allow extra space",
            "Avoid dual-task overload early—progress as tolerated",
        ],
        "core_interventions": [
            {
                "name": "Amplitude training (LSVT BIG-style)",
                "how": "Exaggerated long steps with wide arm swing; cue ‘BIG’ each step",
                "dose": "3×10–15 m, 5–6 days/week; progress speed as safe",
            },
            {
                "name": "Cueing (auditory/visual)",
                "how": "Metronome 90–110 bpm or taped floor lines/laser for step targets",
                "dose": "Use during 10–15 min walks and turn drills",
            },
            {
                "name": "Postural extensors strengthening",
                "how": "Prone/supported back extensions, band rows, hip extensor work",
                "dose": "2–3 sets × 8–12 reps, 3–4 days/week",
            },
            {
                "name": "Balance + turning practice",
                "how": "Weight-shifts, tandem walking, 180° turns in small steps",
                "dose": "10–15 min/session, daily",
            },
            {
                "name": "Freezing strategy",
                "how": "Stop → weight-shift → step over visual line while counting 1–2",
                "dose": "Rehearse 3–5 min before community ambulation",
            },
        ],
    },
    "hemiplegic": {
        "goals": [
            "Restore symmetrical stance and step timing",
            "Facilitate normal kinematics (heel strike, knee control)",
            "Strengthen key weak groups (dorsiflexors, hip flexors/abductors)",
        ],
        "precautions": [
            "Guard against knee hyperextension in stance",
            "Use harness/parallel bars early if balance is low",
        ],
        "core_interventions": [
            {
                "name": "Weight-shift re-education",
                "how": "Mirror-guided shifts onto affected limb between parallel bars",
                "dose": "3×10 slow shifts, daily",
            },
            {
                "name": "Gait phase drills",
                "how": "Step initiation with heel-first contact; controlled knee flexion",
                "dose": "3×10 steps forward/back, daily",
            },
            {
                "name": "Strength: dorsiflexors & hip abductors",
                "how": "Band dorsiflexion; side-lying hip abduction; step-ups",
                "dose": "2–3 sets × 8–12 reps, 3–4 days/week",
            },
            {
                "name": "FES for foot drop (if available)",
                "how": "Stimulate tibialis anterior during swing to ensure toe clearance",
                "dose": "During 10–20 min walking blocks",
            },
            {
                "name": "Orthoses (AFO)",
                "how": "Maintain neutral ankle for clearance and stability",
                "dose": "During ambulation per therapist prescription",
            },
        ],
    },
    "ataxic": {
        "goals": [
            "Enhance trunk control and balance",
            "Improve timing/accuracy of limb movements",
            "Reduce fall risk with deliberate pacing",
        ],
        "precautions": [
            "Wide-base device initially (quad cane/walker)",
            "Avoid fatigue; rest to keep movements accurate",
        ],
        "core_interventions": [
            {
                "name": "Frenkel’s coordination set",
                "how": "Slow, visually guided heel-to-shin, target taps, foot circles",
                "dose": "10–15 min, 5–6 days/week",
            },
            {
                "name": "Static→dynamic balance",
                "how": "Feet together → semi-/full tandem; then line-walk with focus",
                "dose": "10–15 min/session, daily",
            },
            {
                "name": "Core stabilization",
                "how": "Pelvic tilts, dead bug, side planks (as tolerated)",
                "dose": "2–3 sets × 8–12 reps, 3–4 days/week",
            },
            {
                "name": "Sensory weighting",
                "how": "Light ankle/wrist weights or weighted vest for feedback",
                "dose": "Short bouts during gait drills; avoid overloading",
            },
        ],
    },
    "spastic": {
        "goals": [
            "Reduce tone; increase ROM at adductors, hamstrings, calves",
            "Strengthen antagonists (hip abductors, dorsiflexors, extensors)",
            "Normalize step pattern with repetitive gait practice",
        ],
        "precautions": [
            "Slow, sustained stretches; avoid quick ballistic inputs",
            "Consider medical tone management (e.g., Botox) if severe",
        ],
        "core_interventions": [
            {
                "name": "Daily stretching block",
                "how": "Adductors, hamstrings, gastrocs/soleus (with strap/wall)",
                "dose": "3×30–60 s per muscle group, daily",
            },
            {
                "name": "Antagonist strengthening",
                "how": "Band dorsiflexion; hip abduction (bands); bridges",
                "dose": "2–3 sets × 8–12 reps, 3–4 days/week",
            },
            {
                "name": "Body-weight–supported treadmill",
                "how": "Harness-assisted stepping emphasizing hip/knee flexion in swing",
                "dose": "10–20 min blocks, 3–5 days/week",
            },
            {
                "name": "Hydrotherapy (if available)",
                "how": "Warm-water gait and mobility to dampen tone",
                "dose": "15–20 min sessions, 2–3×/week",
            },
        ],
    },
    "antalgic": {
        "goals": [
            "Offload painful limb until irritability drops",
            "Restore pain-free ROM and symmetrical loading",
            "Rebuild local strength and gait pattern",
        ],
        "precautions": [
            "Screen cause (fracture, severe OA, acute injury) and follow MD load limits",
            "Use cane/crutch on opposite side during painful phase",
        ],
        "core_interventions": [
            {
                "name": "Pain modulation",
                "how": "Cryo/heat/TENS per stage; inflammation control per MD",
                "dose": "5–15 min as indicated, multiple times/day early",
            },
            {
                "name": "Graded weight-bearing",
                "how": "Toe-touch → partial → full; biofeedback with scale",
                "dose": "Short bouts hourly progressing to continuous",
            },
            {
                "name": "ROM maintenance",
                "how": "Gentle joint ROM within pain limits; patellar/soft tissue glides",
                "dose": "1–2×/day, 5–10 min",
            },
            {
                "name": "Closed-chain strength",
                "how": "Mini-squats, step-ups, heel raises as pain allows",
                "dose": "2–3 sets × 8–12 reps, 3–4 days/week",
            },
            {
                "name": "Gait re-education",
                "how": "Mirror practice to remove limp once pain <3/10",
                "dose": "5–10 min during walks",
            },
        ],
    },
}


def _extract_categories(rule_label: str | None, rule_hits: Any) -> List[str]:
    text = " ".join([
        (rule_label or ""),
        " ".join(rule_hits if isinstance(rule_hits, (list, tuple)) else [str(rule_hits or "")]),
    ]).lower()
    cats = []
    for key in THERAPY_DB.keys():
        if key in text:
            cats.append(key)
    return sorted(set(cats)) or ["normal/unclear"]


def build_plan(categories: List[str]) -> Dict[str, Any]:
    if categories == ["normal/unclear"]:
        return {
            "summary": "No strong abnormality detected by rules; prioritize general fitness: brisk walking 20–30 min/day, balance circuits 2–3×/wk, global strength (push/pull/hinge/squat) 2–3×/wk.",
            "items": [],
            "plan_markdown": (
                "# General Conditioning Plan\n\n"
                "- Brisk walk 20–30 min daily\n"
                "- Balance circuit (tandem stance, single-leg support near counter) 10–15 min, 2–3×/wk\n"
                "- Strength: 2–3 sets × 8–12 reps of squats, hip hinges, rows, presses, 2–3×/wk\n"
            ),
        }

    lines = ["# Gait Rehab Plan\n"]
    all_items: List[Dict[str, Any]] = []

    for cat in categories:
        db = THERAPY_DB[cat]
        lines.append(f"## {cat.capitalize()} Gait\n")
        lines.append("**Goals**")
        for g in db["goals"]:
            lines.append(f"- {g}")
        lines.append("**Precautions**")
        for p in db["precautions"]:
            lines.append(f"- {p}")
        lines.append("**Core Interventions (with dosage)**")
        for it in db["core_interventions"]:
            all_items.append({"category": cat, **it})
            lines.append(f"- **{it['name']}** — {it['how']} (**Dose:** {it['dose']})")
        lines.append("")

    return {
        "summary": ", ".join([c.capitalize() for c in categories]) + " plan generated",
        "items": all_items,
        "plan_markdown": "\n".join(lines).strip() + "\n",
    }


def recommend_from_feats(feats: Dict[str, Any]) -> Dict[str, Any]:
    rule_label = feats.get("rule_label")
    raw_hits = feats.get("rule_hits", [])
    if isinstance(raw_hits, str):
        rule_hits = [h.strip() for h in raw_hits.split(",") if h.strip()]
    else:
        rule_hits = list(raw_hits) if raw_hits else []
    cats = _extract_categories(rule_label, rule_hits)
    return build_plan(cats)


# -----------------------------
# Batch runner & CSV export
# -----------------------------

def run_batch(input_path, out_csv=None, show=False, draw=False, out_dir=None, print_plan=False):
    # Defensive: if someone passed a list, take first
    if isinstance(input_path, (list, tuple)):
        input_path = input_path[0]

    # Collect video files
    if os.path.isdir(input_path):
        paths = []
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
            paths += glob.glob(os.path.join(input_path, ext))
    elif os.path.isfile(input_path):
        paths = [input_path]
    else:
        raise SystemExit(f"Input not found: {input_path}")

    if not paths:
        raise SystemExit(f"No videos found in: {input_path}")

    print(f"[INFO] Found {len(paths)} video(s) in: {input_path}", flush=True)

    extractor = GaitFeatureExtractor(draw=draw, show=show)
    rows = []

    for i, p in enumerate(sorted(paths), 1):
        print(f"▶ Processing {i}/{len(paths)}: {os.path.basename(p)}", flush=True)
        try:
            feats = extractor.process_video(p)
            feats['video'] = os.path.basename(p)
            rows.append(feats)
            print(f"✔ {os.path.basename(p)} => {feats['rule_label']} | Plan: {feats.get('plan_summary','')}", flush=True)
            # Pretty console
            pretty_print_result(os.path.basename(p), feats)

            # CSV log
            append_to_csv(out_csv, os.path.basename(p), feats)
            if print_plan:
                md = feats.get('plan_markdown', '')
                if md:
                    print("\n----- THERAPY PLAN (Markdown) -----\n")
                    print(md)
                    print("----- END PLAN -----\n")
        except Exception as e:
            print(f"✖ Error on {p}: {e}")

    if out_csv and rows:
        dirn = os.path.dirname(out_csv)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        df = pd.json_normalize(rows)
        df.to_csv(out_csv, index=False)
        print(f"Saved feature+plan table: {out_csv}")

        # Optional: write per-video Markdown rehab plan files
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            for _, row in df.iterrows():
                video = row.get('video', f'row_{_}')
                md = row.get('plan_markdown', '')
                with open(os.path.join(out_dir, f"{os.path.splitext(str(video))[0]}.md"), 'w', encoding='utf-8') as f:
                    f.write(md)
            print(f"Wrote per-video Markdown plans to: {out_dir}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description='Gait feature extraction + rule-based classification + therapy recommendations')
    BASE_DIR = Path(__file__).resolve().parent.parent
    DEFAULT_INPUT = str(BASE_DIR / "data" / "gait_videos")
    DEFAULT_OUT   = str(BASE_DIR / "results" / "gait_features.csv")

    ap.add_argument('--input',
        default=DEFAULT_INPUT,
        help=f'Video file or folder (default: {DEFAULT_INPUT})')

    ap.add_argument('--out_csv',
    default=DEFAULT_OUT,
    help=f'Path to save features CSV (default: {DEFAULT_OUT})')
    ap.add_argument('--print_plan', type=int, default=0,
                help='Print full therapy plan_markdown to console (0/1)')

    ap.add_argument('--show', type=int, default=1, help='Show visualization (0/1)')
    ap.add_argument('--draw', type=int, default=1, help='Draw landmarks on frames (0/1)')
    ap.add_argument('--out_dir', default=None, help='Optional directory to write per-video Markdown rehab plans')
    return ap.parse_args()


def main():
    args = parse_args()
    run_batch(args.input,
          out_csv=args.out_csv,
          show=bool(args.show),
          draw=bool(args.draw),
          out_dir=args.out_dir,
          print_plan=bool(args.print_plan))

def append_to_csv(csv_path, video_name, feats):
    Path(csv_path).parent.mkdir(exist_ok=True)

    row = {
        "video": video_name,
        "label": feats["rule_label"],
        "rule_hits": feats["rule_hits"],
        "therapy_summary": feats["plan_summary"],
        "therapy_markdown": feats["plan_markdown"]
    }

    df = pd.DataFrame([row])

    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)



if __name__ == '__main__':
    main()
