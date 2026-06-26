import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO('1.yaml')
    model = YOLO('runs/DUO.py3.10.cuda12.1/yolov8n-MLCA/weights/best.pt')
    model.val(data=r'MyDataset/DUO-main-4/data.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              iou=0.8,
              )