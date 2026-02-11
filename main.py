import cv2
import base64
import requests
import os
import json
import re
import subprocess
from tqdm import tqdm
from collections import Counter

# ================== CONFIG ==================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-vl:2b"

VIDEO_PATH = "/home/alna/Downloads/a (2)/test_1.webm"
FRAME_DIR = "frames"
FRAME_INTERVAL_SECONDS = 1
RESIZE_WIDTH = 640
OUTPUT_JSON = "analysis_output.json"
# ============================================


# ========== LIVE CAMERA CONFIG ==========
USE_WEBCAM = True       # switch mode
WEBCAM_INDEX = 0          # default webcam
LIVE_DURATION_SEC = 60    # how long to run live analysis
LIVE_FRAME_INTERVAL = 1   # seconds
DISPLAY_FEED = True
# =======================================



# ---------- VIDEO UTILS ----------
def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return round(float(json.loads(result.stdout)["format"]["duration"]), 2)


def validate_video(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print(f" Video validated: {path}")


# ---------- FRAME EXTRACTION (TIME BASED) ----------
def extract_frames(video_path, output_dir, interval_sec):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(" Cannot open video")

    saved = 0
    next_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if current_time >= next_time:
            h, w = frame.shape[:2]
            scale = RESIZE_WIDTH / w
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

            path = f"{output_dir}/frame_{saved:04d}.jpg"
            cv2.imwrite(path, frame)
            saved += 1
            next_time += interval_sec

    cap.release()
    return saved


# ---------- OLLAMA HELPERS ----------
def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def clean_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


def aggression_score(frame):
    score = 0.0
    if frame.get("panic_or_running"):
        score += 0.3
    if frame.get("aggression"):
        score += 0.4
    if frame.get("fight_detected"):
        score += 0.6
    if frame.get("weapons_visible"):
        score += 0.8
    return round(min(score, 1.0), 2)


def analyze_frame(image_path, frame_id, timestamp):
    prompt = f"""
Return ONLY valid JSON. No markdown.

{{
  "frame_id": {frame_id},
  "timestamp_sec": {timestamp},
  "people_count": number,
  "crowd_density": "low" | "medium" | "high",
  "aggression": true | false,
  "fight_detected": true | false,
  "panic_or_running": true | false,
  "suspicious_behavior": true | false,
  "weapons_visible": true | false,
  "confidence": number between 0 and 1,
  "notes": "short observation"
}}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_to_base64(image_path)],
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()

    frame = clean_json(r.json()["response"])
    frame["aggression_score"] = aggression_score(frame)
    return frame


# ---------- SUMMARY ----------
def generate_video_summary(frames):
    people = [f["people_count"] for f in frames]
    densities = [f["crowd_density"] for f in frames]

    summary = {
        "average_people_count": round(sum(people) / len(people), 2),
        "max_people_count": max(people),
        "panic_frames": sum(f["panic_or_running"] for f in frames),
        "fight_frames": sum(f["fight_detected"] for f in frames),
        "weapons_frames": sum(f["weapons_visible"] for f in frames),
        "dominant_crowd_density": Counter(densities).most_common(1)[0][0],
        "overall_behavior": ""
    }

    if summary["fight_frames"] > 0:
        summary["overall_behavior"] = "physical violence detected"
    elif summary["panic_frames"] > len(frames) * 0.4:
        summary["overall_behavior"] = "crowd panic or rapid movement"
    else:
        summary["overall_behavior"] = "normal crowd behavior"

    return summary


# ---------- MAIN PIPELINE ----------
def analyze_video():
    validate_video(VIDEO_PATH)
    duration = get_video_duration(VIDEO_PATH)

    total_frames = extract_frames(
        VIDEO_PATH, FRAME_DIR, FRAME_INTERVAL_SECONDS
    )

    print(f"\n Extracted {total_frames} frames (every {FRAME_INTERVAL_SECONDS}s)\n")

    frames = []

    for i in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        t = i * FRAME_INTERVAL_SECONDS
        frame_path = f"{FRAME_DIR}/frame_{i:04d}.jpg"
        frames.append(analyze_frame(frame_path, i, t))

    video_score = round(
        sum(f["aggression_score"] for f in frames) / len(frames), 2
    )

    output = {
        "video_path": VIDEO_PATH,
        "duration_sec": duration,
        "frame_interval_sec": FRAME_INTERVAL_SECONDS,
        "total_frames_analyzed": len(frames),
        "processing_complete": True,
        "video_aggression_score": video_score,
        "video_risk_level": (
            "low" if video_score < 0.3 else
            "medium" if video_score < 0.6 else
            "high"
        ),
        "video_summary": generate_video_summary(frames),
        "frames": frames
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n Analysis complete. Saved to {OUTPUT_JSON}")

def analyze_live_camera():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("\n Live camera started. Press 'q' to stop.\n")

    frames = []
    start_time = cv2.getTickCount()
    last_analyzed = 0
    frame_id = 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tick_freq = cv2.getTickFrequency()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = (cv2.getTickCount() - start_time) / tick_freq

        # Resize for consistency
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame_resized = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

        if elapsed - last_analyzed >= LIVE_FRAME_INTERVAL:
            temp_path = f"{FRAME_DIR}/live_{frame_id:04d}.jpg"
            os.makedirs(FRAME_DIR, exist_ok=True)
            cv2.imwrite(temp_path, frame_resized)

            try:
                analysis = analyze_frame(temp_path, frame_id, round(elapsed, 2))
                frames.append(analysis)
            except Exception as e:
                print(f"Frame {frame_id} skipped:", e)

            last_analyzed = elapsed
            frame_id += 1

        if DISPLAY_FEED:
            cv2.imshow("Live Camera Feed", frame_resized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if elapsed >= LIVE_DURATION_SEC:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not frames:
        print("No frames analyzed.")
        return

    video_score = round(
        sum(f["aggression_score"] for f in frames) / len(frames), 2
    )

    output = {
        "source": "live_camera",
        "duration_sec": round(elapsed, 2),
        "frame_interval_sec": LIVE_FRAME_INTERVAL,
        "total_frames_analyzed": len(frames),
        "processing_complete": True,
        "video_aggression_score": video_score,
        "video_risk_level": (
            "low" if video_score < 0.3 else
            "medium" if video_score < 0.6 else
            "high"
        ),
        "video_summary": generate_video_summary(frames),
        "frames": frames
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n Live analysis complete. Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    if USE_WEBCAM:
        analyze_live_camera()
    else:
        analyze_video()


