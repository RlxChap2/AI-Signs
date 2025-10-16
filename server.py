from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
from collections import deque

app = Flask(__name__)
CORS(app)

# Mediapipe - إعداد اليدين مرة واحدة فقط
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# لتخزين حركة اليد الأخيرة
hand_history = deque(maxlen=10)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'لم يتم إرسال صورة'}), 400

    # تحويل Base64 إلى صورة
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # تصغير الصورة لتسريع المعالجة
    small_img = cv2.resize(img, (160, int(160 * img.shape[0] / img.shape[1])))

    gesture = "غير معروف"
    coords = []
    wave = False

    # تحليل اليد
    results = hands.process(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lmList = []
        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
            lmList.append((cx, cy))
            coords.append({'x': cx, 'y': cy})

        # رسم اليد على الصورة (اختياري)
        mp_draw.draw_landmarks(
            img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1)
        )

        # التعرف على الأصابع
        fingers = []
        # الإبهام
        fingers.append(1 if (lmList[4][0] < lmList[3][0] if lmList[17][0] > lmList[5][0] else lmList[4][0] > lmList[3][0]) else 0)
        # باقي الأصابع
        tips = [8, 12, 16, 20]
        for tip in tips:
            fingers.append(1 if lmList[tip][1] < lmList[tip - 2][1] else 0)

        # التعرف على الإشارة
        gestures_map = {
            (1,0,0,0,0): "إعجاب 👍",
            (0,1,0,0,0): "إشارة السبابة ☝️",
            (1,1,1,1,1): "يد مفتوحة ✋",
            (0,0,0,0,0): "قبضة ✊"
        }
        gesture = gestures_map.get(tuple(fingers), "غير معروف")

        # التعرف على حركة التلويح
        wrist_x = lmList[0][0]
        hand_history.append(wrist_x)
        if len(hand_history) == hand_history.maxlen:
            diffs = [hand_history[i+1]-hand_history[i] for i in range(len(hand_history)-1)]
            if max(diffs)-min(diffs) > 40:
                wave = True

    return jsonify({
        "gesture": gesture,
        "coords": coords[:5],  # إرسال أول 5 فقط لتقليل البيانات
        "wave": wave
    })


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
