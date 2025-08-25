# app.py
import time
import random
import numpy as np
from threading import Lock

import cv2
import mediapipe as mp
import streamlit as st

# -------------------------------
# Streamlit page setup + styles
# -------------------------------
st.set_page_config(page_title="🖐️ MediaPipe Rock–Paper–Scissors", layout="wide")

CUSTOM_CSS = """
<style>
:root { --bg:#0b1020; --ink:#e9eef7; --muted:#a6b1c9; --acc:#7cc0ff; }
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 800px at 20% 0%, #17213f 0%, #0b1020 50%, #0b1020 100%) !important;
}
[data-testid="stAppViewContainer"] * { color: var(--ink); }
h1, h2, h3, h4 { letter-spacing: .5px; }
.glass-card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
  backdrop-filter: blur(6px);
}
.stat-row {
  display:flex; justify-content:space-between; padding:10px 0;
  border-bottom:1px dashed rgba(255,255,255,0.15);
}
.stat-row:last-child { border-bottom:0; }
.pill {
  display:inline-block; padding:2px 8px; border-radius:999px;
  border:1px solid rgba(255,255,255,.2)
}
.big { font-size: 14px; font-weight: 500; }
.muted { color: var(--muted); }
table.moves { width:100%; border-collapse:collapse; margin-top:10px; }
table.moves th, table.moves td {
  text-align:left; padding:8px 6px; border-bottom:1px solid rgba(255,255,255,0.1);
}
table.moves th { color: var(--muted); font-weight:600; }
.stButton > button, .stNumberInput input {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  color: var(--ink) !important;
  border-radius: 12px !important;
}
.video-frame {
  width: 100%;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.15);
  background: #000;
  object-fit: cover;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------
# MediaPipe setup
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# -------------------------------
# Helpers: drawing (OpenCV)
# -------------------------------
def draw_center_text(img, text, scale=2.0, thickness=3, y_offset=0, color=(255, 255, 255)):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2 + y_offset
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=18):
    x1, y1 = pt1
    x2, y2 = pt2
    w, h = x2 - x1, y2 - y1
    r = min(radius, w // 2, h // 2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, -1)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, -1)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, -1)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, -1)
    alpha = 0.20
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def draw_box_with_label(img, title, label, side="left"):
    h, w = img.shape[:2]
    x = int(w * (0.05 if side == "left" else 0.6))
    y = int(h * 0.2)
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
# Gesture detection
# -------------------------------
def is_finger_up(landmarks, finger_tip_id, finger_pip_id, finger_mcp_id=None):
    tip = landmarks[finger_tip_id]
    pip = landmarks[finger_pip_id]
    if finger_mcp_id:
        mcp = landmarks[finger_mcp_id]
        return tip.y < pip.y and tip.y < mcp.y
    else:
        return tip.y < pip.y

def is_thumb_up(landmarks, handedness_label):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    if handedness_label == "Right":
        horizontal_extended = thumb_tip.x < thumb_ip.x
    else:
        horizontal_extended = thumb_tip.x > thumb_ip.x
    vertical_extended = thumb_tip.y < thumb_mcp.y
    return horizontal_extended or vertical_extended

def classify_gesture(hand_landmarks, handedness_label):
    landmarks = hand_landmarks.landmark
    thumb_up = is_thumb_up(landmarks, handedness_label)
    index_up = is_finger_up(landmarks, 8, 6, 5)
    middle_up = is_finger_up(landmarks, 12, 10, 9)
    ring_up = is_finger_up(landmarks, 16, 14, 13)
    pinky_up = is_finger_up(landmarks, 20, 18, 17)
    fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
    total_up = sum(fingers_up)
    # print(f"Fingers up: T:{thumb_up} I:{index_up} M:{middle_up} R:{ring_up} P:{pinky_up} Total:{total_up}")
    if total_up <= 1:
        return "rock"
    elif total_up >= 4:
        return "paper"
    elif index_up and middle_up and not ring_up and not pinky_up:
        return "scissors"
    elif total_up == 2 and index_up and middle_up:
        return "scissors"
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
        self.history = []
        self.state = "idle"
        self.state_t0 = time.time()

        self.REQUIRED_STABLE_FRAMES = 8
        self.CAPTURE_TIMEOUT = 8.0
        self.COUNTDOWN_SECS = 3
        self.REVEAL_SECS = 2.0
        self.PAUSE_BETWEEN_SECS = 1.5

        self.stable_move = None
        self.stable_count = 0
        self.player_move = None
        self.computer_move = None
        self.last_round_msg = ""
        self._lock = Lock()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not found")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def _go(self, new_state):
        self.state = new_state
        self.state_t0 = time.time()
        if new_state == "capture":
            self.stable_move = None
            self.stable_count = 0
            self.player_move = None
            self.computer_move = None

    def next_frame(self):
        with self._lock:
            ok, frame = self.cap.read()
            if not ok:
                frame = (255 * np.ones((480, 640, 3), dtype=np.uint8))
            frame = cv2.flip(frame, 1)

            if self.state == "idle":
                draw_center_text(frame, "Rock • Paper • Scissors", scale=1.2, thickness=3, y_offset=-120)
                draw_center_text(frame, "Get Ready!", scale=2.0, thickness=6, y_offset=-40, color=(0, 255, 0))
                draw_center_text(frame, f"Round {self.round_num} / {self.total_rounds}",
                                 scale=1.1, thickness=3, y_offset=40)
                self._go("countdown")

            elif self.state == "countdown":
                elapsed = time.time() - self.state_t0
                remaining = self.COUNTDOWN_SECS - int(elapsed)
                remaining = max(1, remaining) if elapsed < self.COUNTDOWN_SECS else 1
                scale = max(2.0, 4.5 - (elapsed * 2.5))
                draw_center_text(frame, str(remaining), scale=scale, thickness=8, color=(0, 255, 0))
                if elapsed >= self.COUNTDOWN_SECS:
                    self._go("capture")

            elif self.state == "capture":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)
                current_guess = None
                if res.multi_hand_landmarks and res.multi_handedness:
                    hand_landmarks = res.multi_hand_landmarks[0]
                    handedness_label = res.multi_handedness[0].classification[0].label
                    current_guess = classify_gesture(hand_landmarks, handedness_label)
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                if current_guess in ("rock", "paper", "scissors"):
                    if self.stable_move == current_guess:
                        self.stable_count += 1
                    else:
                        self.stable_move = current_guess
                        self.stable_count = 1
                else:
                    self.stable_move = None
                    self.stable_count = 0

                cv2.putText(frame, "Show your move!", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

                if current_guess:
                    cv2.putText(frame, f"Detecting: {current_guess} ({self.stable_count}/{self.REQUIRED_STABLE_FRAMES})",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

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
                    if elapsed > self.CAPTURE_TIMEOUT + 1.5:
                        self._go("countdown")

            elif self.state == "reveal":
                draw_box_with_label(frame, "Computer", move_to_label(self.computer_move), side="left")
                draw_box_with_label(frame, "Player", move_to_label(self.player_move), side="right")
                if time.time() - self.state_t0 >= self.REVEAL_SECS:
                    res_winner = winner(self.player_move, self.computer_move)
                    ts = time.strftime("%H:%M:%S")
                    if res_winner == "player":
                        self.player_score += 1
                        self.last_round_msg = f"You win Round {self.round_num}!"
                    elif res_winner == "computer":
                        self.computer_score += 1
                        self.last_round_msg = f"Computer wins Round {self.round_num}!"
                    else:
                        self.last_round_msg = f"Round {self.round_num} is a Draw."
                    # record history tail locally (not persisted to CSV)
                    self.history.append([ts, self.round_num, self.player_move, self.computer_move, res_winner])
                    self.round_num += 1
                    if self.round_num > self.total_rounds:
                        self._go("final")
                    else:
                        self._go("between")

            elif self.state == "between":
                draw_center_text(frame, self.last_round_msg, scale=1.2, thickness=4, y_offset=-30)
                draw_center_text(frame, f"Score  You {self.player_score} : {self.computer_score} Computer",
                                 scale=1.1, thickness=3, y_offset=40)
                if time.time() - self.state_t0 >= self.PAUSE_BETWEEN_SECS:
                    self._go("countdown")

            elif self.state == "final":
                if self.player_score > self.computer_score:
                    msg = f"You won! {self.player_score} / {self.total_rounds}"
                elif self.computer_score > self.player_score:
                    msg = f"Computer won! {self.computer_score} / {self.total_rounds}"
                else:
                    msg = f"Tie game {self.player_score}:{self.computer_score}"
                draw_center_text(frame, "Game Over", scale=2.5, thickness=7, y_offset=-60)
                draw_center_text(frame, msg, scale=1.2, thickness=4, y_offset=20)

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
        (tw, th), _ = cv2.getTextSize(text_center, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
        cv2.putText(frame, text_center, ((w - tw)//2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 2, cv2.LINE_AA)
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
                "history_tail": self.history[-5:]
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

# -------------------------------
# Session state helpers
# -------------------------------
def get_game():
    if "game" not in st.session_state:
        st.session_state.game = RPSGame(total_rounds=st.session_state.get("rounds", 3))
    return st.session_state.game

def stop_game():
    if "running" in st.session_state and st.session_state.running:
        st.session_state.running = False
    if "game" in st.session_state and st.session_state.game is not None:
        st.session_state.game.release()
        del st.session_state.game

# -------------------------------
# UI Header (matches original)
# -------------------------------
with st.container():
    top_cols = st.columns([1, 1])
    with top_cols[0]:
        st.markdown("### 🖐️ MediaPipe Rock–Paper–Scissors")
    with top_cols[1]:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown('<div class="muted" style="margin-bottom:6px;">Rounds</div>', unsafe_allow_html=True)
            rounds = st.number_input("Rounds", min_value=1, max_value=9, value=st.session_state.get("rounds", 3), key="rounds", label_visibility="collapsed")
        with c2:
            restart_clicked = st.button("Restart")
        with c3:
            run_toggle = st.toggle("Start / Stop", value=st.session_state.get("running", False))
            st.session_state.running = run_toggle

if restart_clicked:
    # restart with the chosen number of rounds
    try:
        g = get_game()
        g.restart(st.session_state.rounds)
    except Exception:
        stop_game()
        st.session_state.game = RPSGame(total_rounds=st.session_state.rounds)

# -------------------------------
# Main grid
# -------------------------------
left, right = st.columns([2, 1], gap="medium")

# Video/Card
with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Stats/Card
with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    status_placeholder = st.empty()
    round_placeholder = st.empty()
    score_placeholder = st.empty()
    st.markdown('<div class="muted" style="margin-top:12px;margin-bottom:6px;">Last result</div>', unsafe_allow_html=True)
    last_placeholder = st.empty()
    st.markdown('<div class="muted" style="margin-top:16px;margin-bottom:6px;">Recent moves</div>', unsafe_allow_html=True)
    moves_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

def render_stats(state):
    status_html = f'<div class="stat-row"><span class="muted">Status</span><span class="pill">{ "Finished" if state["is_final"] else "Playing…" }</span></div>'
    round_html  = f'<div class="stat-row"><span class="muted">Round</span><span class="big">{state["round"]} / {state["total_rounds"]}</span></div>'
    score_html  = f'<div class="stat-row"><span class="muted">Score</span><span class="big">You {state["player_score"]} – {state["computer_score"]} CPU</span></div>'
    status_placeholder.markdown(status_html, unsafe_allow_html=True)
    round_placeholder.markdown(round_html, unsafe_allow_html=True)
    score_placeholder.markdown(score_html, unsafe_allow_html=True)
    last_placeholder.markdown(state["last_result"] or "—")
    # table
    rows = state.get("history_tail", [])
    table_html = """
    <table class="moves">
      <thead><tr><th>Time</th><th>Rnd</th><th>Player</th><th>CPU</th><th>Result</th></tr></thead>
      <tbody>
    """
    for ts, rnd, pm, cm, res in rows[::-1]:
        table_html += f"<tr><td>{ts}</td><td>{rnd}</td><td>{pm}</td><td>{cm}</td><td>{res}</td></tr>"
    table_html += "</tbody></table>"
    moves_placeholder.markdown(table_html, unsafe_allow_html=True)

# -------------------------------
# Main loop
# -------------------------------
try:
    if st.session_state.running:
        game = get_game()
        # Continuous update loop (use a small sleep to yield control)
        # Note: Streamlit reruns script on each interaction. We keep this loop tight.
        loop_iterations = 0
        while st.session_state.running:
            frame = game.next_frame()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, use_column_width=True, channels="RGB", output_format="JPEG")
            render_stats(game.get_state())
            time.sleep(0.03)  # ~30 FPS target
            loop_iterations += 1
            # Safety break to avoid extremely long single-run loops on some hosts
            if loop_iterations > 2000:
                break
        # If we broke out because of final state, keep last frame & stats displayed
        render_stats(game.get_state())
    else:
        # Not running: show a placeholder frame / instructions
        # Create a simple black frame with title to match the aesthetic
        frame = (np.zeros((480, 640, 3), dtype=np.uint8))
        draw_center_text(frame, "Rock • Paper • Scissors", scale=1.2, thickness=3, y_offset=-120, color=(200,200,255))
        draw_center_text(frame, "Click Start to play", scale=1.4, thickness=4, y_offset=-30, color=(120, 200, 255))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb, use_column_width=True, channels="RGB", output_format="JPEG")
        # Show stats if a game exists
        if "game" in st.session_state:
            render_stats(st.session_state.game.get_state())
        else:
            render_stats({
                "round": 1, "total_rounds": st.session_state.get("rounds", 3),
                "player_score": 0, "computer_score": 0,
                "last_result": "", "is_final": False, "history_tail": []
            })
finally:
    # If user toggled off, ensure camera is released
    if not st.session_state.get("running", False):
        if "game" in st.session_state and st.session_state.game is not None:
            st.session_state.game.release()
