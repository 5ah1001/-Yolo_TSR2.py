# Yolo_TSR2.py
from ultralytics import YOLO
import multiprocessing
import serial
import cv2
import time


def determine_direction(class_id):
    if class_id == 0:# غيرالاشارات الي هنا ي احمد
        return b'R' # ودا الout put
    elif class_id == 1:
        return b'L'
    elif class_id == 2:
        return b'S'
    else:
        return b'N'


if __name__ == '__main__':
    multiprocessing.freeze_support()

    ser = serial.Serial(
        port='COM5', # ودا ال المخرج بتاعك ي وحش
        baudrate=115200, #دا خليه 9600
        bytesize=8,
        parity='N',
        stopbits=1,
        timeout=1
    )

    model = YOLO(r"C:\Users\ra141\Downloads\Traffic Sign Recognition\best (2).pt") # دا الموديل الي انا هبعتهولك
    cap = cv2.VideoCapture(r"C:\Users\ra141\Downloads\Traffic Sign Recognition\archive (1)\video.mp4") # شيل كل الي جوا الاقواس وحط 0 لو كاميرا داخليه و و1 لو كمرا خارجيه

    prev_direction = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.7)

        current_direction = b'N'
        for result in results:
            if result.boxes:
                class_id = int(result.boxes.cls[0])
                current_direction = determine_direction(class_id)

        if current_direction != prev_direction:
            ser.write(current_direction)
            print(f"Sent: {current_direction}")
            prev_direction = current_direction

        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    ser.close()
    cap.release()
    cv2.destroyAllWindows()