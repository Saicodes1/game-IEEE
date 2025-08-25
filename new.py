import os
import time
import random
import numpy as np  # Missing import
from datetime import datetime
from threading import Lock

from flask import Flask, render_template_string, Response, jsonify, request
import cv2
import mediapipe as mp

# -------------------------------
# Flask
# -------------------------------
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # no caching while you iterate

# -------------------------------
# MediaPipe setup
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# -------------------------------
# Helpers: drawing
# -------------------------------
def draw_center_text(img, text, scale=2.0, thickness=3, y_offset=0, color=(255, 255, 255)):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2 + y_offset
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=18):
    # Basic rounded-rect: draw filled rectangle + four circles + outline
    x1, y1 = pt1
    x2, y2 = pt2
    w, h = x2 - x1, y2 - y1
    r = min(radius, w // 2, h // 2)
    # Fill
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, -1)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, -1)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, -1)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, -1)
    # Transparency
    alpha = 0.20
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Outline
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def draw_box_with_label(img, title, label, side="left"):
    h, w = img.shape[:2]

    # Adjust position to ensure visibility
    if side == "left":
        x = int(w * 0.05)  # Margin from the left
    else:
        x = int(w * 0.6)  # Adjusted margin for better visibility
    y = int(h * 0.2)  # Adjusted position lower on the screen

    # Increase font size for better visibility
    cv2.putText(img, f"{title}: {label}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

def move_to_label(move):
    if move == "rock":
        return "Rock  ROCK"
    if move == "paper":
        return "Hand  PAPER"
    if move == "scissors":
        return "V  SCISSORS"
    return "?"

# -------------------------------
# Improved Gesture Detection
# -------------------------------
def is_finger_up(landmarks, finger_tip_id, finger_pip_id, finger_mcp_id=None):
    """Check if a finger is extended based on landmark positions"""
    tip = landmarks[finger_tip_id]
    pip = landmarks[finger_pip_id]
    
    # For most fingers, check if tip is above PIP
    if finger_mcp_id:
        mcp = landmarks[finger_mcp_id]
        # Additional check: tip should be further from palm than PIP
        return tip.y < pip.y and tip.y < mcp.y
    else:
        return tip.y < pip.y

def is_thumb_up(landmarks, handedness_label):
    """Special logic for thumb detection"""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    
    # For right hand (from camera perspective), thumb extends to the left
    # For left hand, thumb extends to the right
    if handedness_label == "Right":
        # Right hand: thumb tip should be to the left of thumb IP
        horizontal_extended = thumb_tip.x < thumb_ip.x
    else:
        # Left hand: thumb tip should be to the right of thumb IP
        horizontal_extended = thumb_tip.x > thumb_ip.x
    
    # Also check vertical extension
    vertical_extended = thumb_tip.y < thumb_mcp.y
    
    return horizontal_extended or vertical_extended

def classify_gesture(hand_landmarks, handedness_label):
    """Improved gesture classification"""
    landmarks = hand_landmarks.landmark
    
    # Check each finger
    thumb_up = is_thumb_up(landmarks, handedness_label)
    index_up = is_finger_up(landmarks, 8, 6, 5)      # Index finger
    middle_up = is_finger_up(landmarks, 12, 10, 9)   # Middle finger
    ring_up = is_finger_up(landmarks, 16, 14, 13)    # Ring finger
    pinky_up = is_finger_up(landmarks, 20, 18, 17)   # Pinky finger
    
    fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
    total_up = sum(fingers_up)
    
    # Debug info (you can remove this later)
    print(f"Fingers up: T:{thumb_up} I:{index_up} M:{middle_up} R:{ring_up} P:{pinky_up} Total:{total_up}")
    
    # Classification logic
    if total_up <= 1:
        return "rock"
    elif total_up >= 4:
        return "paper"
    elif index_up and middle_up and not ring_up and not pinky_up:
        return "scissors"
    elif total_up == 2 and index_up and middle_up:
        return "scissors"  # More lenient scissors detection
    
    return None

# -------------------------------
# Game rules
# -------------------------------
def decide_computer_move_different(player_move):
    options = ["rock", "paper", "scissors"]
    if player_move in options:
        options.remove(player_move)
    return random.choice(options)

def winner(player, computer):
    if player == computer:
        return "draw"
    if (player == "rock" and computer == "scissors") or \
       (player == "paper" and computer == "rock") or \
       (player == "scissors" and computer == "paper"):
        return "player"
    return "computer"

# -------------------------------
# Game Engine (state machine)
# -------------------------------
class RPSGame:
    """
    States:
      'idle' -> 'countdown' -> 'capture' -> 'reveal' -> ('between' or 'final')
    """
    def __init__(self, total_rounds=3):
        self.total_rounds = total_rounds

        self.player_score = 0
        self.computer_score = 0
        self.round_num = 1
        self.history = []  # Removed CSV-related functionality

        self.state = "idle"
        self.state_t0 = time.time()

        self.REQUIRED_STABLE_FRAMES = 8  # Increased for better stability
        self.CAPTURE_TIMEOUT = 8.0  # Increased timeout
        self.COUNTDOWN_SECS = 3
        self.REVEAL_SECS = 2.0  # Increased reveal time
        self.PAUSE_BETWEEN_SECS = 1.5

        self.stable_move = None
        self.stable_count = 0
        self.player_move = None
        self.computer_move = None
        self.last_round_msg = ""
        self._lock = Lock()

        # Vision with improved settings
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not found")
        
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,  # Increased confidence
            min_tracking_confidence=0.7    # Increased confidence
        )

    # ---------- persistence ----------
    # Removed _save_history method

    # ---------- state helpers ----------
    def _go(self, new_state):
        self.state = new_state
        self.state_t0 = time.time()
        # reset per-state vars
        if new_state == "capture":
            self.stable_move = None
            self.stable_count = 0
            self.player_move = None
            self.computer_move = None

    # ---------- frame pipeline ----------
    def next_frame(self):
        with self._lock:
            print("[DEBUG] Attempting to read frame from camera...")
            ok, frame = self.cap.read()
            if not ok:
                print("[DEBUG] Camera read failed. Creating placeholder frame.")
                # create a placeholder frame if camera hiccups
                frame = (255 * np.ones((480, 640, 3), dtype=np.uint8))
            else:
                print("[DEBUG] Camera frame read successfully.")
            frame = cv2.flip(frame, 1)

            # Start state machine
            print(f"[DEBUG] Current state: {self.state}")
            if self.state == "idle":
                print("[DEBUG] State is 'idle'. Drawing idle screen.")
                draw_center_text(frame, "Rock • Paper • Scissors", scale=1.2, thickness=3, y_offset=-120)
                draw_center_text(frame, "Get Ready!", scale=2.0, thickness=6, y_offset=-40, color=(0, 255, 0))
                draw_center_text(frame, f"Round {self.round_num} / {self.total_rounds}",
                                 scale=1.1, thickness=3, y_offset=40)
                self._go("countdown")

            elif self.state == "countdown":
                print("[DEBUG] State is 'countdown'. Drawing countdown.")
                elapsed = time.time() - self.state_t0
                remaining = self.COUNTDOWN_SECS - int(elapsed)
                remaining = max(1, remaining) if elapsed < self.COUNTDOWN_SECS else 1
                # shrinking effect
                scale = max(2.0, 4.5 - (elapsed * 2.5))
                draw_center_text(frame, str(remaining), scale=scale, thickness=8, color=(0, 255, 0))
                if elapsed >= self.COUNTDOWN_SECS:
                    self._go("capture")

            elif self.state == "capture":
                print("[DEBUG] State is 'capture'. Processing hand landmarks.")
                # Run mediapipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)

                current_guess = None
                if res.multi_hand_landmarks and res.multi_handedness:
                    hand_landmarks = res.multi_hand_landmarks[0]
                    handedness_label = res.multi_handedness[0].classification[0].label
                    current_guess = classify_gesture(hand_landmarks, handedness_label)
                    
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                
                # stability check
                if current_guess in ("rock", "paper", "scissors"):
                    if self.stable_move == current_guess:
                        self.stable_count += 1
                    else:
                        self.stable_move = current_guess
                        self.stable_count = 1
                else:
                    self.stable_move = None
                    self.stable_count = 0

                # HUD
                cv2.putText(frame, "Show your move!", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
                
                # Show current detection
                if current_guess:
                    cv2.putText(frame, f"Detecting: {current_guess} ({self.stable_count}/{self.REQUIRED_STABLE_FRAMES})", 
                               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                
                # Show countdown
                elapsed = time.time() - self.state_t0
                remaining_time = max(0, self.CAPTURE_TIMEOUT - elapsed)
                cv2.putText(frame, f"Time: {remaining_time:.1f}s", (20, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

                if self.stable_count >= self.REQUIRED_STABLE_FRAMES:
                    self.player_move = self.stable_move
                    self.computer_move = decide_computer_move_different(self.player_move)
                    self._go("reveal")

                elif elapsed > self.CAPTURE_TIMEOUT:
                    draw_center_text(frame, "No gesture detected - Try again!", scale=1.2, thickness=4, color=(255, 0, 0))
                    if elapsed > self.CAPTURE_TIMEOUT + 1.5:  # Show message for 1.5s
                        self._go("countdown")  # retry same round

            elif self.state == "reveal":
                print("[DEBUG] State is 'reveal'. Showing results.")
                # Show both choices
                draw_box_with_label(frame, "Computer", move_to_label(self.computer_move), side="left")
                draw_box_with_label(frame, "Player", move_to_label(self.player_move), side="right")
                # Remove result text from the video frame
                # The result will only be shown on the HTML page
                if time.time() - self.state_t0 >= self.REVEAL_SECS:
                    res_winner = winner(self.player_move, self.computer_move)
                    if res_winner == "player":
                        self.player_score += 1
                        self.last_round_msg = f"You win Round {self.round_num}!"
                    elif res_winner == "computer":
                        self.computer_score += 1
                        self.last_round_msg = f"Computer wins Round {self.round_num}!"
                    else:
                        self.last_round_msg = f"Round {self.round_num} is a Draw."

                    self.round_num += 1
                    if self.round_num > self.total_rounds:
                        self._go("final")
                    else:
                        self._go("between")

            elif self.state == "between":
                print("[DEBUG] State is 'between'. Showing between-round screen.")
                draw_center_text(frame, self.last_round_msg, scale=1.2, thickness=4, y_offset=-30)
                draw_center_text(frame, f"Score  You {self.player_score} : {self.computer_score} Computer",
                                 scale=1.1, thickness=3, y_offset=40)
                if time.time() - self.state_t0 >= self.PAUSE_BETWEEN_SECS:
                    self._go("countdown")

            elif self.state == "final":
                print("[DEBUG] State is 'final'. Showing final screen.")
                # Final banner
                if self.player_score > self.computer_score:
                    msg = f"You won! {self.player_score} / {self.total_rounds}"
                elif self.computer_score > self.player_score:
                    msg = f"Computer won! {self.computer_score} / {self.total_rounds}"
                else:
                    msg = f"Tie game {self.player_score}:{self.computer_score}"

                draw_center_text(frame, "Game Over", scale=2.5, thickness=7, y_offset=-60)
                draw_center_text(frame, msg, scale=1.2, thickness=4, y_offset=20)

            print("[DEBUG] Returning processed frame.")
            # Top overlay bar: round & score (nice and subtle)
            self._draw_top_bar(frame)
            return frame

    def _draw_top_bar(self, frame):
        h, w = frame.shape[:2]
        bar_h = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        text_left = f"Round: {min(self.round_num, self.total_rounds)} / {self.total_rounds}"
        text_center = "Rock • Paper • Scissors"
        text_right = f"Score: You {self.player_score} – {self.computer_score} CPU"
        cv2.putText(frame, text_left, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
        # center text
        (tw, th), _ = cv2.getTextSize(text_center, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
        cv2.putText(frame, text_center, ((w - tw)//2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 2, cv2.LINE_AA)
        # right text
        (tw, _), _ = cv2.getTextSize(text_right, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
        cv2.putText(frame, text_right, (w - tw - 12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 2, cv2.LINE_AA)

    def get_state(self):
        with self._lock:
            summary = {
                "round": min(self.round_num, self.total_rounds),
                "total_rounds": self.total_rounds,
                "player_score": self.player_score,
                "computer_score": self.computer_score,
                "last_result": self.last_round_msg,
                "is_final": self.state == "final",
                "history_tail": self.history[-5:]  # last few moves not yet flushed
            }
            return summary

    def restart(self, total_rounds=None):
        with self._lock:
            if total_rounds:
                self.total_rounds = int(total_rounds)
            self.player_score = 0
            self.computer_score = 0
            self.round_num = 1
            self.last_round_msg = ""
            self.history.clear()
            self._go("idle")

    def release(self):
        try:
            self.hands.close()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass

# Global game
game = None

# -------------------------------
# Flask routes
# -------------------------------
PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MediaPipe RPS — Flask</title>
  <style>
    :root { --bg:#0b1020; --ink:#e9eef7; --muted:#a6b1c9; --acc:#7cc0ff; }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 24px; background: radial-gradient(1200px 800px at 20% 0%, #17213f 0%, #0b1020 50%, #0b1020 100%);
      color: var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue";
    }
    h1 { margin: 0 0 10px; font-size: 28px; letter-spacing: .5px;}
    .wrap { max-width: 1100px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 18px; align-items: start; }
    .card {
      background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
      border-radius: 18px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.25);
      backdrop-filter: blur(6px);
    }
    #video {
      width: 100%; aspect-ratio: 16/9; border-radius: 14px; border: 1px solid rgba(255,255,255,0.15);
      background: #000; display: block; object-fit: cover;
    }
    .stat { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed rgba(255,255,255,0.15); }
    .stat:last-child { border-bottom: 0; }
    .big { font-size: 10px; font-weight: 500; }
    .muted { color: var(--muted); }
    .row { display: flex; gap: 10px; align-items: center; }
    button {
      appearance: none; border: 1px solid rgba(255,255,255,0.12); color: var(--ink);
      background: rgba(255,255,255,0.06); border-radius: 12px; padding: 10px 14px; font-size: 14px;
      cursor: pointer;
    }
    button:hover { background: rgba(255,255,255,0.1); }
    input[type="number"] {
      width: 90px; border-radius: 10px; padding: 8px 10px; background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.14); color: var(--ink);
    }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid rgba(255,255,255,0.1); }
    th { color: var(--muted); font-weight: 600; }
    .pill { display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid rgba(255,255,255,.2)}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="row" style="justify-content: space-between; margin-bottom: 12px;">
      <h1>🖐️ MediaPipe Rock–Paper–Scissors</h1>
      <div class="row">
        <label for="rounds" class="muted">Rounds</label>
        <input id="rounds" type="number" min="1" max="9" value="3" />
        <button id="btnStart">Restart</button>
      </div>
    </div>
    <div class="grid">
      <div class="card">
        <img id="video" src="/video_feed" alt="Video stream" />
      </div>
      <div class="card">
        <div class="stat"><span class="muted">Status</span><span id="status" class="pill">Playing…</span></div>
        <div class="stat"><span class="muted">Round</span><span id="round" class="big">1 / 3</span></div>
        <div class="stat"><span class="muted">Score</span><span id="score" class="big">You 0 – 0 CPU</span></div>
        <div style="margin-top: 12px;">
          <div class="muted" style="margin-bottom:6px;">Last result</div>
          <div id="lastResult" style="font-weight:700;">&nbsp;</div>
        </div>
        <div style="margin-top: 16px;">
          <div class="muted" style="margin-bottom:6px;">Recent moves</div>
          <table>
            <thead><tr><th>Time</th><th>Rnd</th><th>Player</th><th>CPU</th><th>Result</th></tr></thead>
            <tbody id="histBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
  <script>
    const roundEl = document.getElementById('round');
    const scoreEl = document.getElementById('score');
    const lastEl  = document.getElementById('lastResult');
    const statusEl = document.getElementById('status');
    const histBody = document.getElementById('histBody');
    const btnStart = document.getElementById('btnStart');
    const inRounds = document.getElementById('rounds');

    async function poll(){
      try{
        const res = await fetch('/game_state',{cache:'no-store'});
        const j = await res.json();
        roundEl.textContent = `${j.round} / ${j.total_rounds}`;
        scoreEl.textContent = `You ${j.player_score} – ${j.computer_score} CPU`;
        lastEl.textContent = j.last_result || '—';
        statusEl.textContent = j.is_final ? 'Finished' : 'Playing…';
        statusEl.style.borderColor = j.is_final ? 'rgba(124,192,255,0.5)' : 'rgba(255,255,255,0.2)';

        // update tail
        histBody.innerHTML = '';
        (j.history_tail || []).slice().reverse().forEach(row=>{
          const [ts, rnd, pm, cm, res] = row;
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${ts}</td><td>${rnd}</td><td>${pm}</td><td>${cm}</td><td>${res}</td>`;
          histBody.appendChild(tr);
        });
      }catch(e){ /* ignore */ }
    }
    setInterval(poll, 500);
    poll();

    btnStart.addEventListener('click', async ()=>{
      const r = parseInt(inRounds.value || '3', 10);
      await fetch('/restart?rounds='+Math.max(1, Math.min(9, r)), {method:'POST'});
      setTimeout(poll, 300);
      // Refresh the <img> stream to avoid stale buffer in some browsers:
      const v = document.getElementById('video');
      v.src = '/video_feed?ts=' + Date.now();
    });
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(PAGE_HTML)

def mjpeg_stream():
    global game
    while True:
        try:
            print("[DEBUG] Fetching next frame...")
            frame = game.next_frame()
            print("[DEBUG] Frame fetched successfully.")
            # Encode as JPEG
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                print("[DEBUG] JPEG encoding failed.")
                continue
            print("[DEBUG] JPEG encoding succeeded.")
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        except Exception as e:
            print(f"[ERROR] Exception in mjpeg_stream: {e}")

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/game_state")
def game_state():
    return jsonify(game.get_state())

@app.route("/restart", methods=["POST"])
def restart():
    req_rounds = int(request.args.get("rounds", "3"))
    game.restart(req_rounds)
    return jsonify({"ok": True})

# -------------------------------
# Entry
# -------------------------------
if __name__ == "__main__":
    try:
        game = RPSGame(total_rounds=3)
        app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
    finally:
        if game:
            game.release()