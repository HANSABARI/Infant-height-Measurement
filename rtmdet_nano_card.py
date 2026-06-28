# rtmdet_nano_card.py
_base_ = 'mmdet::rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

# 1. 경로 설정 (사진의 폴더 구조와 일치시킴)
data_root = 'dataset/card_detection-2/'
classes = ('card',)  # 우리가 찾을 객체 이름

# 2. 모델 설정 (기본 80개 클래스를 1개로 변경)
model = dict(
    bbox_head=dict(num_classes=1)
)

# 3. 데이터셋 설정
train_dataloader = dict(
    batch_size=8, # Mac 로컬 테스트용으로 무리 안 가게 살짝 낮춤
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='test/_annotations.coco.json',   # test 폴더의 json
        data_prefix=dict(img='test/')             # test 폴더의 이미지
    )
)

test_evaluator = dict(ann_file=data_root + 'test/_annotations.coco.json')  # test 용으로 분리
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')


# 4. 학습 횟수(Epoch) 및 저장 경로 설정
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=100)
work_dir = './app/models/card_model_v2' # 학습된 모델이 저장될 곳

# 5. Pre-trained 모델 가중치 로드 (처음부터가 아니라 똑똑한 상태에서 시작)
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'