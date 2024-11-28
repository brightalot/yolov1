import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from model_496 import Yolov1
from utils import (
    non_max_suppression,
    plot_image,
    load_checkpoint,
    get_bboxes,
    mean_average_precision
)

# 모델 설정 및 가중치 로드
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL_FILE = "sgd_64_1000.pth.tar"  # 훈련된 모델 파일 경로
IMAGE_PATH = "sample_img.jpeg"
# 이미지 크기 조정 및 텐서 변환
transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def load_image_from_path(img_path):
    image = Image.open(img_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)  # 모델 입력 형태로 변환
    return image  # 변환된 이미지 반환

def main():
    # 모델 불러오기
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model)
    
    
    # 지정된 경로에서 이미지 로드
    image = load_image_from_path(IMAGE_PATH)
    
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
    output_image_path = os.path.join("result/sgd_16_135", os.path.basename("test"))
    plot_image(image[0].permute(1, 2, 0).to("cpu"), filtered_bboxes, labels, scores, output_image_path)
    print(f"Result saved to {output_image_path}")

if __name__ == "__main__":
    main()