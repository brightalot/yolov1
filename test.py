"""
Pascal VOC 데이터셋을 사용하여 학습된 Yolo 모델을 테스트하기 위한 파일
"""

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
    plot_image,
)
from train import DEVICE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY

# 하이퍼파라미터 등 설정
BATCH_SIZE = 64
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL_FILE = "sgd_64_1000.pth.tar"
IMG_DIR = "data/data/images"
LABEL_DIR = "data/data/labels"

# 여러 변환을 적용하는 클래스 정의
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

# 이미지 크기 조정과 텐서 변환을 적용
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

# 메인 함수 정의
def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    # load_checkpoint(torch.load(LOAD_MODEL_FILE), model, None)
    # load_checkpoint(torch.load(LOAD_MODEL_FILE), model, None, load_optimizer=False)
    #optimizer 를 설정해줬어야..
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)


    # 테스트 데이터셋 정의
    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    # 테스트 데이터로더 생성
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    # 모델 평가 모드로 전환
    model.eval()

    pred_boxes, target_boxes = get_bboxes(
        test_loader, model, iou_threshold=0.5, threshold=0.4
    )

    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    print(f"Test mAP: {mean_avg_prec}")

    # 예측 결과 시각화
    for idx in range(3):  # 테스트 데이터 중 3개 이미지 시각화
        image, _ = test_dataset[idx+37]
        image = image.to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            pred = model(image)
            bboxes = get_bboxes(
                [(image, pred)], model, iou_threshold=0.5, threshold=0.4
                )[0]
            
            labels = [box[1] for box in bboxes]  # 각 박스에서 클래스 인덱스 추출
            scores = [box[2] for box in bboxes]  # 각 박스에서 확률 점수 추출
            print("labels", labels)
            print("scores", scores)
            # print("bboxes", bboxes)
            # [[0, 18.0, 0.601652204990387, 0.506049394607544, 0.5142343640327454, 0.5951494574546814, 0.4710894823074341]]
            
            # plot_image(image[0].permute(1, 2, 0).to("cpu"), bboxes)

            # Extract only the x, y, w, h values for plotting
            # filtered_bboxes = [box[3:7] for box in bboxes]
            # filtered_bboxes = [[box[i] for i in range(2, 6)] for box in bboxes]
            filtered_bboxes = [[box[3], box[4], box[5], box[6]] for box in bboxes if len(box) >= 7]
            print("Filtered Boxes:", filtered_bboxes)

            plot_image(image[0].permute(1, 2, 0).to("cpu"), filtered_bboxes, labels, scores, "result/sgd_64_1000_{}.png".format(idx))



if __name__ == "__main__":
    main()