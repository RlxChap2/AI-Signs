from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

app = Flask(__name__)

# إعداد Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# متغير لحفظ آخر إحداثيات المعصم للكشف عن التلويح
hand_history = deque(maxlen=10)

# دالة توليد الفريمات من الكاميرا
def generate_frames():
    cap = cv2.VideoCapture(0)
    gesture = "لا توجد إشارة"

    while True:
        success, img = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        h, w, c = img.shape

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((cx, cy))

                if lmList:
                    # ---------------------------
                    # 👇 منطق تحليل الإشارة
                    # ---------------------------
                    fingers = []

                    # الإبهام (محور أفقي)
                    if lmList[4][0] > lmList[3][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # باقي الأصابع (محور عمودي)
                    tips = [8, 12, 16, 20]
                    for tip in tips:
                        fingers.append(1 if lmList[tip][1] < lmList[tip - 2][1] else 0)

                    # تحديد الإشارة من النمط
                    patterns = {
                        (0, 0, 0, 0, 0): "قبضة ✊",
                        (1, 1, 1, 1, 1): "يد مفتوحة ✋",
                        (0, 1, 0, 0, 0): "سبابة ☝️",
                        (1, 0, 0, 0, 0): "إعجاب 👍",
                        (0, 1, 1, 0, 0): "نصر ✌️",
                        (0, 1, 1, 1, 1): "أربع أصابع 🖐️",
                        (1, 0, 0, 0, 1): "اتصال 🤙",
                    }

                    gesture = patterns.get(tuple(fingers), "غير معروف")

                    # التلويح (Wave Detection)
                    wrist_x = lmList[0][0]
                    hand_history.append(wrist_x)

                    if len(hand_history) == hand_history.maxlen:
                        diffs = [hand_history[i + 1] - hand_history[i] for i in range(len(hand_history) - 1)]
                        if max(diffs) - min(diffs) > 50:
                            gesture = "👋 تلويح"

                    # رسم النقاط والخطوط
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # عرض الإشارة الحالية على الكاميرا
        cv2.putText(img, f"الإشارة: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # تحويل الصورة للفيديو البثّي
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
