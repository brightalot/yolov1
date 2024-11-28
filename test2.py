import torch
import torchvision.transforms as transforms
from model import Yolov1
from dataset import VOCDataset
from utils import (
    get_bboxes,
    load_checkpoint,
    plot_image,
    mean_average_precision,
)
from train import DEVICE
LOAD_MODEL_FILE = "sgd_16_135_2.pth.tar"

# 이미지 크기 조정과 텐서 변환을 적용
# 여러 변환을 적용하는 클래스 정의
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img = t(img)  # 이미지만 변환 적용
        return img, bboxes  # 변환된 이미지와 원본 바운딩 박스 반환


# 이미지 크기 조정과 텐서 변환을 적용
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


# 메인 함수 정의
def main():
    # 모델 불러오기
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = None  # 불필요한 optimizer 설정 제거
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # 임의의 이미지 불러오기
    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir="data/data/images", label_dir="data/data/labels"
    )
    
    model.eval()

    idx = 1  # 테스트할 임의의 인덱스
    image, _ = test_dataset[idx]
    image = image.to(DEVICE).unsqueeze(0)

    with torch.no_grad():
        # 예측 수행
        pred = model(image)
        # 예측된 박스와 실제 박스, 첫 번째 이미지의 예측 박스 가져오기
        pred_boxes, target_boxes = get_bboxes(
            [(image, pred)], model, iou_threshold=0.5, threshold=0.4
        )
        bboxes = pred_boxes  # 첫 번째 이미지의 예측 박스

        # 클래스와 점수 추출
        labels = [box[1] for box in bboxes]
        scores = [box[2] for box in bboxes]
        # 예측 박스 좌표 추출 (x, y, w, h)
        filtered_bboxes = [[box[3], box[4], box[5], box[6]] for box in bboxes if len(box) >= 7]
        
        # 예측 결과 시각화 및 저장
        plot_image(image[0].permute(1, 2, 0).to("cpu"), filtered_bboxes, labels, scores, "result/sgd_64_1000_sample.png")

        # mAP 계산 및 출력
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Mean Average Precision (mAP): {mean_avg_prec}")


if __name__ == "__main__":
    main()
