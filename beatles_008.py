import cv2
import mediapipe as mp
import pygame
import random
import time
import numpy as np
import math

# Mediapipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ウェブカメラのキャプチャ
cap = cv2.VideoCapture(0)
frame_rate = 30
cap.set(cv2.CAP_PROP_FPS, frame_rate)

# Pygameを初期化して音声を再生するための準備
pygame.mixer.init()

# ウィンドウの表示用フォント
font = cv2.FONT_HERSHEY_SIMPLEX

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center = (width // 2, height // 2)  # 画像の中心座標

class Ball:
    def __init__(self, speed, color, sound=None):
        # 初期位置を決定(画面の中心)
        self.speed = speed
        self.x, self.y = width // 2, height // 2
        self.x_speed = 0
        self.y_speed = 0
        # 玉の初期ベクトルをランダムに決定
        while self.x_speed == 0 and self.y_speed == 0:
            self.x_speed = random.choice([-self.speed, 0, self.speed])
            self.y_speed = random.choice([-self.speed, 0, self.speed])
        if self.x_speed == 0 or self.y_speed == 0: # 調整
            self.x_speed = self.x_speed * math.sqrt(2)
            self.y_speed = self.y_speed * math.sqrt(2)
        # BGR
        # 水色, 黄緑, オレンジ
        self.color = color
        self.size = 20
        self.sound = sound
        self.isAudioPlayed = False

    def move(self):
        self.x += self.x_speed
        self.y += self.y_speed

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, self.color, -1)

    def play_sound(self):
        if self.sound is not None and self.isAudioPlayed == False:
            self.sound.play()
            self.isAudioPlayed = True

BPM = 60
hihat_beats = [1, 1, 1, 1, 1, 1, 1, 1]
snare_beats = [0, 0, 1, 0, 0, 0, 1, 0]
kick_beats = [1, 0, 0, 0, 1, 1, 0, 0]

ball_speed = 3
interval = 60 / BPM # 秒数ごとにBallを作成する間隔
hihat = []
snare = []
kick = []

# 音源ファイルのパス
hihat_sound_file = pygame.mixer.Sound("hihat.mp3")
snare_sound_file = pygame.mixer.Sound("snare.mp3")
kick_sound_file = pygame.mixer.Sound("kick.mp3")

counter = 0
window_name = "window"
# Mediapipeのモデルを初期化
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

# キャプチャの開始
start_time = time.time()
while cap.isOpened():
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time >= interval:
        if hihat_beats[counter % 8]:
            hihat.append(Ball(ball_speed, (193, 185, 90), hihat_sound_file))
        if snare_beats[counter % 8]:
            snare.append(Ball(ball_speed, (98, 193, 90), snare_sound_file))
        if kick_beats[counter % 8]:
            kick.append(Ball(ball_speed, (90, 124, 193), kick_sound_file))
        start_time = current_time
        counter += 1

    for drum_part in [hihat, snare, kick]:
        for ball in drum_part:
            ball.move()
            # 透明な円の部分を通過した場合に音源を再生
            distance = int(math.sqrt((ball.x - center[0])**2 + (ball.y - center[1])**2))
            if radius <= distance and distance < radius + thickness:
                ball.play_sound()

    ret, frame = cap.read()
    if not ret:
        break

    # 映像を反転させる
    frame = cv2.flip(frame, 1)

    # フレームをRGBに変換
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False

    # Mediapipeで手と姿勢を検出
    results = holistic.process(frame)

    # 手と姿勢の座標を取得
    left_hand_landmarks = results.left_hand_landmarks
    right_hand_landmarks = results.right_hand_landmarks
    pose_landmarks = results.pose_landmarks

    # 手の検出がある場合、左手の丸を描画
    if left_hand_landmarks:
        left_hand_x = [landmark.x for landmark in left_hand_landmarks.landmark]
        left_hand_y = [landmark.y for landmark in left_hand_landmarks.landmark]
        left_hand_center_x = sum(left_hand_x) / len(left_hand_x)
        left_hand_center_y = sum(left_hand_y) / len(left_hand_y)
        thickness = 10
        cv2.circle(frame, (int(left_hand_center_x * frame.shape[1]), int(left_hand_center_y * frame.shape[0])), 30, (0, 200, 0), thickness)

    # 手の検出がある場合、右手の丸を描画
    if right_hand_landmarks:
        right_hand_x = [landmark.x for landmark in right_hand_landmarks.landmark]
        right_hand_y = [landmark.y for landmark in right_hand_landmarks.landmark]
        right_hand_center_x = sum(right_hand_x) / len(right_hand_x)
        right_hand_center_y = sum(right_hand_y) / len(right_hand_y)
        thickness = 10
        cv2.circle(frame, (int(right_hand_center_x * frame.shape[1]), int(right_hand_center_y * frame.shape[0])), 30, (0, 0, 255), thickness)

    # 足の検出がある場合、左足の丸を描画
    if pose_landmarks:
        left_foot_x = [pose_landmarks.landmark[27].x, pose_landmarks.landmark[29].x, pose_landmarks.landmark[31].x]
        left_foot_y = [pose_landmarks.landmark[27].y, pose_landmarks.landmark[29].y, pose_landmarks.landmark[31].y]
        left_foot_center_x = sum(left_foot_x) / len(left_foot_x)
        left_foot_center_y = sum(left_foot_y) / len(left_foot_y)
        thickness = 10
        cv2.circle(frame, (int(left_foot_center_x * frame.shape[1]), int(left_foot_center_y * frame.shape[0])), 30, (255, 0, 0), thickness)

    # 足の検出がある場合、右足の丸を描画
    if pose_landmarks:
        right_foot_x = [pose_landmarks.landmark[28].x, pose_landmarks.landmark[30].x, pose_landmarks.landmark[32].x]
        right_foot_y = [pose_landmarks.landmark[28].y, pose_landmarks.landmark[30].y, pose_landmarks.landmark[32].y]
        right_foot_center_x = sum(right_foot_x) / len(right_foot_x)
        right_foot_center_y = sum(right_foot_y) / len(right_foot_y)
        thickness = 10
        cv2.circle(frame, (int(right_foot_center_x * frame.shape[1]), int(right_foot_center_y * frame.shape[0])), 30, (255, 255, 0), thickness)


    # 円を描画
    radius = 200  # 円の半径
    thickness = 50  # 円の線の太さ
    
    # 玉を描画
    for drum_part in [hihat, snare, kick]:
        for ball in drum_part:
            ball.draw(frame)

    for i in range(16):
        angle_start = i * np.pi / 8 + np.pi / 16  # 開始角度
        angle_end = (i + 1) * np.pi / 8 + np.pi / 16 # 終了角度

        overlay = frame.copy()
        if i % 2 == 0:
            cv2.ellipse(overlay, center, (radius, radius), 0, int(angle_start * 180 / np.pi), int(angle_end * 180 / np.pi), (255, 255, 255), thickness)
        else:
            cv2.ellipse(overlay, center, (radius, radius), 0, int(angle_start * 180 / np.pi), int(angle_end * 180 / np.pi), (0, 0, 200), thickness)

        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # 画面外に消えた要素を削除
    hihat = [ball for ball in hihat if 0 <= ball.x < width and 0 <= ball.y < height]
    snare = [ball for ball in snare if 0 <= ball.x < width and 0 <= ball.y < height]
    kick = [ball for ball in kick if 0 <= ball.x < width and 0 <= ball.y < height]


    # フレームを表示
    cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
