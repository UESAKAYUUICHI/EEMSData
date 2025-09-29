from ultralytics import YOLO
import os, shutil

DATASET_DIR = 'D:/pythonProject/AbnormalEnvMonitor/EEMSData'


TRAIN_IMAGES = os.path.join(DATASET_DIR, 'train', 'images')
VAL_IMAGES   = os.path.join(DATASET_DIR, 'valid', 'images')

# data_yaml = f"""
# train: {TRAIN_IMAGES}
# val:   {VAL_IMAGES}
# nc: 2
# names: [smoke, fire]
# """
# with open('data.yaml', 'w') as f:
#     f.write(data_yaml)

model = YOLO(r"D:\pythonProject\AbnormalEnvMonitor\EEMSData\yolov8n.pt")


model.train(
    data=os.path.join(DATASET_DIR, 'data.yaml'),
    epochs=300,
    imgsz=896,
    batch=8,
    optimizer='AdamW',
    lr0=1e-3,
    lrf=0.2,
    weight_decay=1e-4,
    momentum=0.937,
    warmup_epochs=10,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,
    augment=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10, translate=0.1, scale=0.5, shear=2.0,
    perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, mixup=0.5, cutmix=0.2,
    copy_paste=0.0,
    amp=True,
    patience=9999,
    project='fire-smoke',
    name='run_enhanced_v2',
    exist_ok=True,
    save=True,
    save_period=1,
    plots=True
)


best_pt = os.path.join('fire-smoke', 'run_enhanced_v2', 'weights', 'best.pt')
if os.path.exists(best_pt):
    shutil.copy(best_pt, './best.pt')
    print("✅ best.pt copied to current working directory")
else:
    print("⚠️ best.pt not found—check for errors")