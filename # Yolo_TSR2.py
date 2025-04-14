# Yolo_TSR
from ultralytics import YOLO
import multiprocessing

if __name__ == '_main_':
    multiprocessing.freeze_support()


model = YOLO(r"C:\Users\Eng.Sahlool\Desktop\ان شاء الله اخر تعديل\best (2).pt")

results = model(source=0, show=True, save=True)