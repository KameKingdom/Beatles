import cv2
import mediapipe as mp
import pygame
import math
import numpy as np

def calculate_angle(a, b, c):
    """
    3点の座標を使用して角度を計算する関数
    """
    angle_rad = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# MediapipeのPoseモジュールを初期化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Webカメラから映像を取得
cap = cv2.VideoCapture(0)

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

# ウィンドウのサイズを取得
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
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

        # 左腕の角度を取得
        if results.pose_landmarks is not None:
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # 左腕が曲げられたらhihatを再生
            if angle < 90:
                # hihatが再生中でない場合に再生開始
                if not is_hihat_playing and not hihat_played:
                    pygame.mixer.music.load('hihat.mp3')
                    pygame.mixer.music.play()
                    is_hihat_playing = True
                    is_snare_playing = False
                    hihat_played = True
                    cv2.putText(image, 'hihat.mp3', (20, 50), font, 1, (0, 255, 0), 2)
            else:
                # 左腕が曲げられていない場合は再生中フラグをリセット
                is_hihat_playing = False
                hihat_played = False

        # 右腕の角度を取得
        if results.pose_landmarks is not None:
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # 右腕が曲げられたらsnareを再生
            if angle < 90:
                # snareが再生中でない場合に再生開始
                if not is_snare_playing and not snare_played:
                    pygame.mixer.music.load('snare.mp3')
                    pygame.mixer.music.play()
                    is_snare_playing = True
                    is_hihat_playing = False
                    snare_played = True
                    cv2.putText(image, 'snare.mp3', (20, 100), font, 1, (0, 255, 0), 2)
            else:
                # 右腕が曲げられていない場合は再生中フラグをリセット
                is_snare_playing = False
                snare_played = False

        # ウィンドウに映像を表示
        # cv2.circle(image, (width // 2, height // 2), 50, (255, 0, 0), -1)  # 水色の円を描画
        cv2.imshow('Skeleton Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
