import cv2
import os
import dlib
from datetime import datetime

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

filter_data = [
    ("Crown", "crown.png", "crown_icon.png"),
    ("Glasses", "glasses.png", "glasses_icon.png"),
    ("Moustache", "moustache.png", "moustache_icon.png"),
    ("Clown Nose", "clown_nose.png", "clown_nose_icon.png"),
    ("Dog Ears", "dog_nose.png", "dog_nose_icon.png"),
]

filters = [(name, cv2.imread(f_img, cv2.IMREAD_UNCHANGED), cv2.imread(i_img)) for name, f_img, i_img in filter_data]
current_filter_idx = None

icon_size = 80
icon_margin = 20
icon_positions = []

os.makedirs("captured_photos", exist_ok=True)

def mouse_click(event, x, y, flags, param):
    global current_filter_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (ix, iy) in enumerate(icon_positions):
            if ix < x < ix + icon_size and iy < y < iy + icon_size:
                current_filter_idx = i

cap = cv2.VideoCapture(0)
cv2.namedWindow("Face Filter App")
cv2.setMouseCallback("Face Filter App", mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    overlay = frame.copy()
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if current_filter_idx is not None:
        filter_name, f_img, _ = filters[current_filter_idx]

        for (x, y, fw, fh) in faces:
            if filter_name == "Glasses":
                x1, y1 = x, y + int(fh / 4.5)
                width = fw
            elif filter_name == "Moustache":
                x1, y1 = x + int(fw / 4), y + int(fh * 0.65)
                width = int(fw / 2)
            elif filter_name == "Clown Nose":
                x1 = x + int(fw / 8)
                y1 = y + int(fh / 4.5)
                width = int(fw * 0.8)

                scale = width / f_img.shape[1]
                resized_filter = cv2.resize(f_img, None, fx=scale, fy=scale)
                fh_s, fw_s = resized_filter.shape[:2]

                x2, y2 = x1 + fw_s, y1 + fh_s
                if x1 >= 0 and y1 >= 0 and x2 <= overlay.shape[1] and y2 <= overlay.shape[0]:
                    alpha_s = resized_filter[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(3):
                        overlay[y1:y2, x1:x2, c] = (
                                alpha_s * resized_filter[:, :, c] + alpha_l * overlay[y1:y2, x1:x2, c]
                        )

                nose_center_x = x + fw // 2
                nose_center_y = y + int(fh * 0.55)
                nose_radius = int(fw * 0.06)
                cv2.circle(overlay, (nose_center_x, nose_center_y), nose_radius, (0, 0, 255), -1)
                continue

            elif filter_name == "Dog Ears":
                x1, y1 = x, y - int(fh / 4.5)
                width = fw
            elif filter_name == "Crown":
                x1, y1 = x, y - int(fh / 1.8)
                width = fw
            else:
                continue

            scale = width / f_img.shape[1]
            resized_filter = cv2.resize(f_img, None, fx=scale, fy=scale)
            fh, fw = resized_filter.shape[:2]
            x2, y2 = x1 + fw, y1 + fh
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue

            alpha_s = resized_filter[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(3):
                overlay[y1:y2, x1:x2, c] = (alpha_s * resized_filter[:, :, c] +
                                            alpha_l * overlay[y1:y2, x1:x2, c])

        cv2.putText(overlay, f"Filter: {filter_name}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    icon_y = h - icon_size - 10
    icon_positions.clear()
    for i, (name, _, icon_img) in enumerate(filters):
        x = icon_margin + i * (icon_size + icon_margin)
        icon_img = cv2.resize(icon_img, (icon_size, icon_size))
        overlay[icon_y:icon_y + icon_size, x:x + icon_size] = icon_img
        icon_positions.append((x, icon_y))

    cv2.imshow("Face Filter App", overlay)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') or key == ord('C'):
        filename = f"captured_photos/photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, overlay)
        print(f"Photo saved to {filename}")
    elif key == 27:
        break
cap.release()
cv2.destroyAllWindows()