import torch
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from model import Yolov1
from utils import (
    non_max_suppression,
    plot_image,
    load_checkpoint,
    get_bboxes,
    mean_average_precision
)
from dataset import VOCDataset

# 모델 설정 및 가중치 로드
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL_FILE = "4096_sgd_16_135.pth.tar"  # 훈련된 모델 파일 경로
CSV_FILE = "data/train.csv"  # 이미지 경로가 저장된 CSV 파일 경로
IMG_DIR = "data/data/images"  # 이미지가 저장된 디렉토리 경로

# 이미지 크기 조정 및 텐서 변환
transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def load_image_from_csv(index):
    """ CSV 파일에서 이미지 경로를 불러와 이미지를 로드하고 변환 """
    df = pd.read_csv(CSV_FILE)
    img_path = os.path.join(IMG_DIR, df.iloc[index, 0])  # CSV 파일에서 이미지 경로 읽기
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0).to(DEVICE)  # 모델 입력 형태로 변환
    return image, img_path  # 이미지와 경로 반환

def main(index):
    # 모델 불러오기
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model)
    
    # CSV 파일에서 이미지 로드
    image, img_path = load_image_from_csv(index)
    
    model.eval()  # 평가 모드로 설정
    with torch.no_grad():
        predictions = model(image)
    
    # Non-Max Suppression 적용
    bboxes = get_bboxes([(image, predictions)], model, iou_threshold=0.5, threshold=0.4)[0]
    
    # 클래스 레이블과 점수 추출
    labels = [box[1] for box in bboxes]
    scores = [box[2] for box in bboxes]
    
    # 좌표 변환 후 시각화
    filtered_bboxes = [[box[3], box[4], box[5], box[6]] for box in bboxes if len(box) >= 7]
    
    pred_boxes, target_boxes = get_bboxes([(image, predictions)], model, iou_threshold=0.5, threshold=0.4)

    # mAP 계산
    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    print(f"Mean Average Precision (mAP): {mean_avg_prec}")

    # 결과 시각화 및 이미지 저장
    output_image_path = os.path.join("result/4096_sgd_16_135", os.path.basename(img_path))
    plot_image(image[0].permute(1, 2, 0).to("cpu"), filtered_bboxes, labels, scores, output_image_path)
    print(f"Result saved to {output_image_path}")

if __name__ == "__main__":
    main(0)
