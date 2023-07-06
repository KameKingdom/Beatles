import cv2
import mediapipe as mp
import pygame
import random
import time

# Webカメラから映像を取得
cap = cv2.VideoCapture(0)

# MediapipeのPoseモジュールを初期化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Pygameを初期化して音声を再生するための準備
pygame.mixer.init()

# 音源の再生状態を管理するフラグ
is_snare_playing = False
is_hihat_playing = False

# 音源再生フラグを管理する変数
snare_played = False
hihat_played = False

# ウィンドウの表示用フォント
font = cv2.FONT_HERSHEY_SIMPLEX

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center = (width // 2, height // 2)  # 画像の中心座標

class Ball:
    def __init__(self, speed, color, sound_file=None):
        # 初期位置を決定(画面の中心)
        self.speed = speed
        self.x, self.y = width // 2, height // 2
        self.x_speed = 0
        self.y_speed = 0
        # 玉の初期ベクトルをランダムに決定
        self.x_speed = random.choice([-self.speed, self.speed])
        self.y_speed = random.choice([-self.speed, self.speed])
        # BGR
        # 水色, 黄緑, オレンジ
        self.color = color
        self.size = 20
        self.sound_file = sound_file

    def move(self):
        self.x += self.x_speed
        self.y += self.y_speed

    def draw(self, image):
        cv2.circle(image, (self.x, self.y), self.size, self.color, -1)

    def play_sound(self):
        if self.sound_file is not None:
            pygame.mixer.music.load(self.sound_file)
            pygame.mixer.music.play()

    def remove(self):
        # 適当な位置に移動して非表示にする
        self.x = -100
        self.y = -100

ball_speed = 3
BPM = 138
interval = 60 / BPM # 秒数ごとにBallを作成する間隔
hihat_beats = [1, 1, 1, 1, 1, 1, 1, 1]
snare_beats = [0, 0, 1, 0, 0, 0, 1, 0]
kick_beats = [1, 0, 0, 0, 1, 1, 0, 0]
hihat = []
snare = []
kick = []

# 音源ファイルのパス
hihat_sound_file = "hihat.mp3"
snare_sound_file = "snare.mp3"
kick_sound_file = "kick.mp3"

counter = 0
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
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
                if ball.x >= center[0] - radius and ball.x <= center[0] + radius and ball.y >= center[1] - radius and ball.y <= center[1] + radius:
                    ball.play_sound()

        ret, frame = cap.read()

        # 映像を反転させる
        frame = cv2.flip(frame, 1)

        # BGR画像をRGB画像に変換
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipeで骨格検出を実行
        results = pose.process(image)

        # 検出結果を描画
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 玉を描画
        for drum_part in [hihat, snare, kick]:
            for ball in drum_part:
                ball.draw(image)

        # 円を描画
        radius = 200  # 円の半径
        thickness = 50  # 円の線の太さ
        overlay = image.copy()
        cv2.circle(overlay, center, radius, (255,255,255), thickness)
        alpha = 0.5
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # 画面外に消えた要素を削除
        hihat = [ball for ball in hihat if 0 <= ball.x < width and 0 <= ball.y < height]
        snare = [ball for ball in snare if 0 <= ball.x < width and 0 <= ball.y < height]
        kick = [ball for ball in kick if 0 <= ball.x < width and 0 <= ball.y < height]

        # ウィンドウのサイズをディスプレイのサイズに合わせる
        cv2.namedWindow('Skeleton Detection', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Skeleton Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # ウィンドウに映像を表示
        cv2.imshow('Skeleton Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
