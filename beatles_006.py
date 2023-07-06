import cv2
import mediapipe as mp

# Mediapipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ウェブカメラのキャプチャ
cap = cv2.VideoCapture(0)

# Mediapipeのモデルを初期化
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# キャプチャの開始
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # フレームをRGBに変換
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Mediapipeで手と姿勢を検出
    results = holistic.process(image)

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
        cv2.circle(frame, (int(left_hand_center_x * frame.shape[1]), int(left_hand_center_y * frame.shape[0])), 30, (0, 255, 0), thickness)

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

    # フレームを表示
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
