from ultralytics import YOLO


def train_model():
    model = YOLO("yolo8m.pt")

    results = model.train(
        data="dataset_r/card_detection-2/data.yaml",
        epochs=50,
        imgsz=640,
        device="mps",
        batch=16,

        # 👇 여기에 경로 설정 옵션을 추가합니다! 👇
        project="app/models",  # 이 폴더 안에 저장해라 (없으면 자동으로 만듦)
        name="card_model_v2"  # 세부 폴더 이름은 이걸로 해라
    )


if __name__ == '__main__':
    train_model()