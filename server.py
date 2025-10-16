from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
from collections import deque

# إنشاء تطبيق Flask بسيط
app = Flask(__name__)
CORS(app)  # للسماح بالاتصال من المتصفح بدون مشاكل

# مكتبة Mediapipe الخاصة باكتشاف اليدين
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# لتخزين حركة اليد الأخيرة (للتعرف على التلويح)
hand_history = deque(maxlen=10)

# الصفحة الرئيسية
@app.route('/')
def index():
    return render_template("index.html")


# المسار المسؤول عن تحليل الصورة القادمة من الكاميرا
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'لم يتم إرسال صورة'}), 400

    # تحويل الصورة من Base64 إلى مصفوفة NumPy قابلة للمعالجة
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # تصغير الصورة لتسريع المعالجة
    small_img = cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1])))

    # المتغيرات الأساسية
    gesture = "غير معروف"
    coords = []
    wave = False

    # تحليل اليد باستخدام Mediapipe
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:
        # نحول الصورة إلى RGB لأن Mediapipe يعمل بهذا النظام اللوني
        results = hands.process(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB))

        # إذا تم اكتشاف يد
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # تخزين جميع النقاط (المفاصل)
                lmList = []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    lmList.append((cx, cy))
                    coords.append({'x': cx, 'y': cy})

                # رسم خطوط اليد والمفاصل
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=5),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # تحديد الأصابع المرفوعة
                fingers = []
                # الإبهام
                if lmList[17][0] > lmList[5][0]:
                    fingers.append(1 if lmList[4][0] < lmList[3][0] else 0)
                else:
                    fingers.append(1 if lmList[4][0] > lmList[3][0] else 0)

                # باقي الأصابع (المؤشر، الوسطى، البنصر، الخنصر)
                tips = [8, 12, 16, 20]
                for tip in tips:
                    fingers.append(1 if lmList[tip][1] < lmList[tip - 2][1] else 0)

                # التعرف على نوع الإشارة
                if fingers == [1, 0, 0, 0, 0]:
                    gesture = "إعجاب 👍"
                elif fingers == [0, 1, 0, 0, 0]:
                    gesture = "إشارة السبابة ☝️"
                elif fingers == [1, 1, 1, 1, 1]:
                    gesture = "يد مفتوحة ✋"
                elif fingers == [0, 0, 0, 0, 0]:
                    gesture = "قبضة ✊"
                else:
                    gesture = "غير معروف"

                # التعرف على حركة التلويح (wave)
                wrist_x = lmList[0][0]
                hand_history.append(wrist_x)

                if len(hand_history) == hand_history.maxlen:
                    # نحسب فرق الحركة في محور X
                    diffs = [hand_history[i + 1] - hand_history[i] for i in range(len(hand_history) - 1)]
                    if max(diffs) - min(diffs) > 40:
                        wave = True

    # تحويل الصورة مرة أخرى إلى Base64 لإرسالها للواجهة
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    img_data_url = f"data:image/jpeg;base64,{img_b64}"

    # إرسال النتائج
    return jsonify({
        "gesture": gesture,
        "coords": coords,
        "wave": wave,
        "image": img_data_url
    })


# تشغيل السيرفر
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)