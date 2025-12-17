import os
import torch
from ultralytics import YOLO

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Используемое устройство: {device}")

model = YOLO("runs/detect/yolov10_m1_final_run/weights/best.pt")

print("\n--- Запуск трекинга с BoT-SORT ---")
model.track(
    source="video.mp4",
    conf=0.25,
    iou=0.45,
    tracker="botsort.yaml",
    show=False,
    save=True,
    name="botsort",
    device=device
)

print("\n--- Запуск трекинга с ByteTrack ---")
model.track(
    source="video.mp4",
    conf=0.25,
    iou=0.45,
    tracker="bytetrack.yaml",
    show=False,
    save=True,
    name="bytetrack",
    device=device 
)