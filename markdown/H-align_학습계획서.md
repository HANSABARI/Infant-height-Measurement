# H-ALIGN AI 모델 학습 계획서 v1.1

**작성일**: 2026-03-09  
**변경**: v1.0 → v1.1 (전체 파이프라인 RTM/OpenMMLab 생태계 통일)  
**프로젝트**: Grow Up MVP · 한사바리 프로젝트  
**라이선스**: 전 모델 Apache 2.0 (상용화 자유)

---

## 1. 변경 요약 (v1.0 → v1.1)

| 항목 | v1.0 | v1.1 |
|------|------|------|
| 카드 검출 | OpenCV 컨투어 → RF-DETR | **RTMDet-nano** (커스텀 학습) |
| 사람 검출 | RTMDet-nano (pretrained) | **RTMDet-nano** (영유아 파인튜닝) |
| 포즈 추정 | RTMPose-WholeBody-l | **RTMPose-WholeBody-l** (영유아 파인튜닝) |
| 프레임워크 | 혼합 (Ultralytics + MMPose) | **OpenMMLab 단일 생태계** |
| 라이선스 | AGPL 혼재 ⚠️ | **전부 Apache 2.0** ✅ |

**RTM 통일 이유**: 카드 검출, 사람 검출, 포즈 추정 3개 모델이 같은 프레임워크(MMEngine + MMCV)를 쓰면 의존성 관리, config 패턴, 학습 파이프라인, ONNX 변환이 전부 동일한 방식으로 통일된다. 초기 러닝커브는 있지만 한 번 익히면 3개 모델에 모두 적용 가능하다.

---

## 2. 파이프라인 모델 매핑

```
입력: 탑뷰 사진 1장 (아이 + 참조 카드)
│
├─── Track A ─────────────────────┐
│    RTMDet-nano (card class)     │
│    → 카드 bbox → 4점 좌표       │
│    → 가로세로 px → px_per_cm    │
│                                 │
├─── Track B ─────────────────────┐
│    RTMDet-nano (person class)   │
│    → 아이 bbox                  │
│         │                       │
│         ▼                       │
│    RTMPose-WholeBody-l          │
│    → 133 키포인트 좌표 (px)      │
│    → 세그먼트 합산 → total_px   │
│                                 │
└─── 최종 산출 (수학) ────────────┘
     height_cm = total_px ÷ px_per_cm
     ± 오차 마진 (신뢰도 기반)
```

| 단계 | 모델 | 프레임워크 | 라이선스 | 파인튜닝 |
|------|------|-----------|---------|---------|
| 카드 검출 | RTMDet-nano (1 class: card) | MMDetection 3.x | Apache 2.0 ✅ | ✅ 커스텀 학습 |
| 사람 검출 | RTMDet-nano (1 class: person) | MMDetection 3.x | Apache 2.0 ✅ | ✅ 영유아 파인튜닝 |
| 포즈 추정 | RTMPose-WholeBody-l (133kp) | MMPose 1.x | Apache 2.0 ✅ | ✅ 영유아 파인튜닝 |
| 키 산출 | 순수 Python 계산 | — | — | 보정계수 튜닝만 |

---

## 3. OpenMMLab 프레임워크 가이드

### 3-1. 아키텍처 개요

```
┌─────────────────────────────────────────────┐
│                 MMEngine                     │
│        (학습 루프, 로깅, 체크포인트)            │
├─────────────────────────────────────────────┤
│                   MMCV                       │
│     (CV 연산, 이미지 처리, CUDA ops)          │
├──────────────────┬──────────────────────────┤
│   MMDetection    │       MMPose             │
│   (RTMDet)       │   (RTMPose)              │
│   카드/사람 검출   │   포즈 추정               │
├──────────────────┴──────────────────────────┤
│                MMDeploy                      │
│        (ONNX/TensorRT 변환, 서빙)            │
└─────────────────────────────────────────────┘
```

### 3-2. 버전 호환 테이블 (검증 완료 조합)

```
Python          3.8 ~ 3.10 (3.10 권장)
PyTorch         2.0 ~ 2.1 + CUDA 11.8
MMEngine        >= 0.9.0
MMCV            >= 2.0.1 (mmcv, not mmcv-lite)
MMDetection     >= 3.1.0
MMPose          >= 1.3.0
MMDeploy        >= 1.3.0 (ONNX 변환 시)

⚠️ 핵심 규칙:
  mmdet 3.x ↔ mmpose 1.x ↔ mmcv 2.x (반드시 세대 일치)
  mmdet 2.x ↔ mmpose 0.x ↔ mmcv 1.x (구버전, 사용 금지)
```

### 3-3. 설치 스크립트 (RunPod / Colab 공용)

```bash
#!/bin/bash
# h-align-env-setup.sh
# RunPod PyTorch 2.1 + CUDA 11.8 템플릿 기준

# 1. conda 환경 (RunPod는 base에서 바로 가능)
# conda create -n halign python=3.10 -y && conda activate halign

# 2. PyTorch (이미 설치되어 있으면 스킵)
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 3. OpenMMLab 코어
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"

# 4. Detection + Pose (소스 설치 권장 - config 수정 용이)
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && pip install -e . && cd ..

git clone https://github.com/open-mmlab/mmpose.git
cd mmpose && pip install -e . && cd ..

# 5. 배포 도구 (ONNX 변환용, 나중에)
# pip install mmdeploy mmdeploy-runtime-gpu onnxruntime-gpu

# 6. 추론 전용 경량 라이브러리 (rtmlib)
pip install rtmlib

# 7. 검증
python -c "
import mmengine; print(f'MMEngine: {mmengine.__version__}')
import mmcv;     print(f'MMCV:     {mmcv.__version__}')
import mmdet;    print(f'MMDet:    {mmdet.__version__}')
import mmpose;   print(f'MMPose:   {mmpose.__version__}')
print('All OK!')
"
```

### 3-4. Config 시스템 이해

OpenMMLab의 config는 **Python 파일 기반 상속 구조**다. H-ALIGN에서 수정할 부분만 이해하면 된다.

```
configs/
├── _base_/                          # 공통 베이스
│   ├── default_runtime.py           # 로깅, 체크포인트 주기
│   └── datasets/
│       └── coco_wholebody.py        # 데이터셋 경로, 파이프라인
│
├── rtmdet/                          # 검출 모델 configs
│   └── rtmdet_nano_8xb32_card.py   # ← 우리가 만들 카드 검출 config
│
└── wholebody_2d_keypoint/           # 포즈 모델 configs  
    └── rtmpose/
        └── coco-wholebody/
            └── rtmpose-l_infant.py  # ← 우리가 만들 영유아 포즈 config
```

**config 상속 패턴 (핵심만):**

```python
# rtmdet_nano_8xb32_card.py — 카드 검출 커스텀 config

_base_ = [
    '../_base_/default_runtime.py',      # 기본 런타임 설정 상속
]

# === 수정할 부분만 오버라이드 ===

# 모델: RTMDet-nano, 클래스 1개(card)
model = dict(
    type='RTMDet',
    backbone=dict(type='CSPNeXt', deepen_factor=0.33, widen_factor=0.25),
    neck=dict(type='CSPNeXtPAFPN', in_channels=[64, 128, 256], out_channels=64),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=1,           # ← card 1개 클래스
        in_channels=64,
    ),
)

# 데이터
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='CocoDataset',
        data_root='data/card_detection/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=dict(classes=('card',)),   # ← 클래스명
    ),
)

val_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root='data/card_detection/',
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=dict(classes=('card',)),
    ),
)

# 학습 설정
train_cfg = dict(max_epochs=100, val_interval=10)
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05))

# pretrained 체크포인트에서 시작
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_nano_8xb32-300e_coco/rtmdet_nano_8xb32-300e_coco_20220903_112922-0e9e8888.pth'
```

### 3-5. Registry 시스템 핵심

모든 컴포넌트를 문자열 이름으로 등록/조회하는 패턴이다. config에서 `type='RTMDet'`이라고 쓰면 Registry에서 해당 클래스를 찾아서 인스턴스화한다.

```python
# 흔한 에러와 대응
KeyError: 'RTMDet is not in the mmdet::model registry'
# → mmdet이 import 안 됨. pip install -e . 재확인

KeyError: 'CSPNeXt is not in the mmdet::backbone registry'  
# → config에 _scope_='mmdet' 누락

RuntimeError: 'TopdownPoseEstimator' is not registered
# → mmpose가 import 안 됨. 설치 확인

# 디버깅 팁: Registry 내용 확인
from mmdet.registry import MODELS
print(MODELS.module_dict.keys())  # 등록된 모든 모델 이름 출력
```

### 3-6. 데이터셋 포맷 (COCO Format)

RTMDet과 RTMPose 모두 **COCO JSON 포맷**을 사용한다.

**카드 검출용 (bbox만):**

```json
{
  "images": [
    {"id": 1, "file_name": "img_001.jpg", "height": 1280, "width": 960}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [120, 450, 280, 176],
      "area": 49280,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "card"}
  ]
}
```

**사람 검출용 (bbox만):**

```json
{
  "categories": [
    {"id": 1, "name": "person"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [50, 100, 400, 600],
      "area": 240000,
      "iscrowd": 0
    }
  ]
}
```

**포즈 추정용 (bbox + 133 keypoints):**

```json
{
  "categories": [
    {
      "id": 1,
      "name": "person",
      "keypoints": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_big_toe", "left_small_toe", "left_heel",
        "right_big_toe", "right_small_toe", "right_heel"
      ],
      "skeleton": [[0,1], [0,2], [1,3], [2,4], ...]
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [50, 100, 400, 600],
      "area": 240000,
      "iscrowd": 0,
      "num_keypoints": 23,
      "keypoints": [
        215, 130, 2,
        220, 125, 2,
        210, 125, 2,
        ...
      ]
    }
  ]
}
```

keypoints 배열 구조: `[x1, y1, v1, x2, y2, v2, ...]` — visibility: 0=미라벨, 1=가려짐, 2=보임

**라벨링 도구 추천:**

| 도구 | 용도 | 장점 |
|------|------|------|
| **CVAT** (Intel) | bbox + keypoint | 웹 기반, COCO JSON 내보내기 지원, 무료 |
| **Label Studio** | bbox + keypoint | 셀프호스팅, 커스텀 라벨 탬플릿 |
| **Roboflow** | bbox 위주 | 라벨링 + augmentation + COCO 내보내기 올인원 |

> **133kp 전체 라벨링은 비현실적.** 영유아 커스텀 데이터에서는 키 측정에 필요한 핵심 23개 키포인트만 라벨링하고, 나머지(손가락, 얼굴 세부)는 0(미라벨)으로 둔다. RTMPose는 부분 라벨링을 허용한다.

### 3-7. 학습 실행 방식

```bash
# ========== 검출 모델 학습 (RTMDet) ==========
cd mmdetection

# 학습 시작
python tools/train.py configs/rtmdet/rtmdet_nano_card.py \
    --work-dir work_dirs/card_det

# 이어서 학습 (체크포인트에서 재개)
python tools/train.py configs/rtmdet/rtmdet_nano_card.py \
    --work-dir work_dirs/card_det \
    --resume

# 평가
python tools/test.py configs/rtmdet/rtmdet_nano_card.py \
    work_dirs/card_det/best_coco_bbox_mAP_epoch_80.pth


# ========== 포즈 모델 학습 (RTMPose) ==========
cd mmpose

# 학습 시작
python tools/train.py configs/wholebody_2d_keypoint/rtmpose/rtmpose-l_infant.py \
    --work-dir work_dirs/infant_pose

# 평가
python tools/test.py configs/wholebody_2d_keypoint/rtmpose/rtmpose-l_infant.py \
    work_dirs/infant_pose/best_coco_wholebody_AP_epoch_100.pth


# ========== ONNX 변환 (MMDeploy) ==========
# 검출 모델
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmdet/detection_onnxruntime_static.py \
    configs/rtmdet/rtmdet_nano_card.py \
    work_dirs/card_det/best.pth \
    test_image.jpg \
    --work-dir onnx_models/card_det

# 포즈 모델
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmpose/pose-detection_onnxruntime_static.py \
    configs/wholebody_2d_keypoint/rtmpose/rtmpose-l_infant.py \
    work_dirs/infant_pose/best.pth \
    test_image.jpg \
    --work-dir onnx_models/infant_pose
```

---

## 4. 모델별 파인튜닝 계획

### 4-1. RTMDet-nano: 카드 검출

| 항목 | 계획 |
|------|------|
| Base model | RTMDet-nano (COCO pretrained) |
| 커스텀 클래스 | card (1 class) |
| 데이터 목표 | 학습 1,500장 + 검증 300장 |
| 학습 환경 | Kaggle T4 (무료) |
| 학습 시간 | ~3시간 (100 epoch) |
| 목표 mAP | > 0.95 |
| 비용 | **$0** |

**데이터 확보:**

| 출처 | 수량 | 방법 |
|------|------|------|
| Roboflow "credit card" 데이터셋 | 300~600장 | 다운로드 → COCO JSON 변환 |
| 직접 촬영 | 500장 | 다양한 배경(바닥, 이불, 매트)에 카드 배치, 탑뷰 |
| Albumentations augmentation | 원본의 3배 | 회전, 밝기, 블러, 원근 변환 |

카드 촬영 시 **반드시 아이(인형 대용 가능)와 함께** 배치하여 실제 사용 환경을 재현한다.

### 4-2. RTMDet-nano: 사람(영유아/유치원) 검출

| 항목 | 계획 |
|------|------|
| Base model | RTMDet-nano (COCO pretrained, person class) |
| 커스텀 클래스 | person (1 class, 영유아 특화) |
| 우선순위 | 1순위: 0~24개월 영유아 (누운 탑뷰) / 2순위: 25~84개월 유치원 (서있는/앉은) |
| 학습 환경 | Kaggle T4 → RunPod RTX 4090 |
| 비용 | **$0~10** |

**Phase 1: COCO pretrained 그대로 테스트 (비용 $0)**

COCO에 이미 person 클래스가 있으므로, 먼저 pretrained 모델로 영유아 탑뷰 사진에서 검출이 되는지 테스트한다. 검출 실패율이 높으면 Phase 2로 진행.

**Phase 2: 영유아 특화 파인튜닝**

| 출처 | 수량 | 설명 |
|------|------|------|
| COCO 2017 (아이 포함 이미지 필터링) | ~5,000장 | pycocotools로 person bbox 중 작은(영유아) 것 추출 |
| 자체 촬영 (탑뷰 누운 영유아) | 500장 | 보호자 협력 또는 인형 활용 |
| SyRIP 데이터셋 (bbox 활용) | ~2,400장 | 영아 bbox 어노테이션 활용 |
| BabyView 데이터셋 | 선별 | 6개월~5세 egocentric 비디오에서 프레임 추출 |

### 4-3. RTMPose-WholeBody-l: 영유아 포즈 추정 ⭐ 핵심

| 항목 | 계획 |
|------|------|
| Base model | RTMPose-WholeBody-l (COCO-WholeBody + UBody pretrained) |
| 키포인트 | 133개 중 키 측정 핵심 23개 우선 라벨링 |
| 우선순위 | 1순위: 0~24개월 누운 영유아 / 2순위: 25~84개월 서있는 유치원 |
| 학습 환경 | RunPod RTX 4090 ($0.34/hr) |
| 학습 시간 | 24~48시간 |
| 비용 | **$8~80** (반복 포함) |

**키 측정 핵심 23개 키포인트 (라벨링 우선순위):**

```
[필수 - 키 측정 직접 사용] 12개
nose(0), left_shoulder(5), right_shoulder(6),
left_hip(11), right_hip(12),
left_knee(13), right_knee(14),
left_ankle(15), right_ankle(16),
left_big_toe(17), right_big_toe(20),
left_heel(19), right_heel(22)

[보조 - head_top 추정용] 5개  
left_eye(1), right_eye(2),
left_ear(3), right_ear(4),
+ face landmark 중 이마 상단 몇 개

[보조 - 자세 판단용] 6개
left_elbow(7), right_elbow(8),
left_wrist(9), right_wrist(10),
left_small_toe(18), right_small_toe(21)
```

**RTMPose 파인튜닝 config:**

```python
# rtmpose-l_infant_wholebody.py

_base_ = ['../../../_base_/default_runtime.py']

# pretrained에서 시작 (COCO-WholeBody 학습 완료된 체크포인트)
load_from = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-wholebody-256x192-4d3e73dd_20230728.pth'

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    head=dict(
        type='RTMCCHead',
        in_channels=1024,
        out_channels=133,          # WholeBody 133 유지
        input_size=(192, 256),
        in_featuremap_size=(6, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        loss=dict(type='KLDiscretLoss', use_target_weight=True, beta=10., label_softmax=True),
    ),
)

# 학습 설정 (파인튜닝용 — 낮은 lr, 적은 epoch)
train_cfg = dict(max_epochs=100, val_interval=10)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),  # 파인튜닝: 낮은 lr
)

# 단일 GPU 배치 사이즈 (RTX 4090 24GB 기준)
train_dataloader = dict(
    batch_size=64,               # 원본 8×A100은 256, 단일 GPU는 64
    num_workers=4,
    dataset=dict(
        type='CocoWholeBodyDataset',
        data_root='data/infant_pose/',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
    ),
)

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='CocoWholeBodyDataset',
        data_root='data/infant_pose/',
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
    ),
)
```

---

## 5. 데이터셋 확보 전체 계획

### 5-1. 1순위: 영유아 (0~24개월)

| 데이터셋 | 유형 | 규모 | 용도 | 접근 |
|---------|------|------|------|------|
| **SyRIP** (Northeastern Univ.) | 실제 + 합성 | ~4,400장 | 포즈 + 검출 | 이메일 요청 (ostadabbas@northeastern.edu) |
| **COCO-WholeBody** | 전연령 | 200K+ | 포즈 pretrained 기반 | 무료 다운로드 |
| **babyPose** (Salesi Hospital) | 조산아 depth | 16K 프레임 | 벤치마크 (비상업) | Zenodo 무료 |
| **자체 촬영** | 탑뷰 누운 영유아 | 목표 500장 | 카드+검출+포즈 | 보호자 동의하에 |
| **합성 데이터** | SMPL-Infant 렌더링 | 5,000장 | 포즈 augmentation | GPU $2~5 |

### 5-2. 2순위: 유치원 (25~84개월, 약 2~7세)

| 데이터셋 | 유형 | 규모 | 용도 | 접근 |
|---------|------|------|------|------|
| **COCO 2017** (아이 필터링) | 서있는/걷는 아이 | ~5,000장 추출 | 검출 + 포즈 | 무료 |
| **Halpe** | 26kp 전연령 | 수만 장 | 포즈 보조 | 무료 |
| **UBody** | WholeBody | 수만 장 | 포즈 보조 | 무료 |
| **BabyView** | 6개월~5세 egocentric | 493시간 비디오 | 프레임 추출 | 연구용 공개 |
| **자체 촬영** | 서있는 유치원생 | 목표 300장 | 카드+검출+포즈 | 보호자 동의 |

### 5-3. 카드 검출 데이터

| 출처 | 규모 | 포맷 |
|------|------|------|
| Roboflow Universe "credit card" | 300~600장 | COCO JSON 내보내기 |
| 직접 촬영 (카드 + 아이/인형) | 500장 | CVAT → COCO JSON |
| Augmentation (Albumentations) | 원본 ×3 | 스크립트로 자동 생성 |
| **합계** | ~3,000장 | |

### 5-4. 데이터 변환 스크립트

Roboflow 등에서 다른 포맷으로 받은 데이터를 COCO JSON으로 변환:

```python
# roboflow_to_coco.py
# Roboflow에서 "COCO" 포맷으로 내보내면 그대로 사용 가능
# 만약 YOLO txt 포맷으로 받았다면:

import json
import os
from PIL import Image

def yolo_to_coco(yolo_dir, output_json, class_names=['card']):
    """YOLO txt → COCO JSON 변환"""
    images, annotations = [], []
    ann_id = 1

    img_dir = os.path.join(yolo_dir, 'images')
    lbl_dir = os.path.join(yolo_dir, 'labels')

    for img_id, fname in enumerate(sorted(os.listdir(img_dir)), 1):
        img_path = os.path.join(img_dir, fname)
        w, h = Image.open(img_path).size
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        txt_path = os.path.join(lbl_dir, fname.rsplit('.', 1)[0] + '.txt')
        if not os.path.exists(txt_path):
            continue

        with open(txt_path) as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls) + 1,
                    "bbox": [round(x, 1), round(y, 1), round(bw * w, 1), round(bh * h, 1)],
                    "area": round(bw * w * bh * h, 1),
                    "iscrowd": 0,
                })
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i+1, "name": n} for i, n in enumerate(class_names)],
    }
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"Converted {len(images)} images, {len(annotations)} annotations")
```

---

## 6. 추론 (서빙) 전략

학습은 MMDetection/MMPose로 하되, **서빙은 rtmlib + ONNX로 경량화**한다.

```python
# h_align_inference.py — FastAPI 서버에서 사용할 추론 코드
# MMPose 의존성 없이 ONNX만으로 추론

from rtmlib import Body, Wholebody
import cv2
import numpy as np

class HAlignPipeline:
    def __init__(self):
        # 포즈 추정 (rtmlib = ONNX 추론, MMPose 불필요)
        self.pose = Wholebody(
            mode='balanced',
            backend='onnxruntime',
            device='cpu',   # 서버: 'cuda'
        )
        # 카드 검출은 별도 ONNX 모델 로드
        # self.card_detector = ort.InferenceSession('card_det.onnx')

    def measure(self, img: np.ndarray) -> dict:
        # 1. 포즈 추정 (133 keypoints)
        keypoints, scores = self.pose(img)

        # 2. 세그먼트 합산 (순수 수학)
        # ...

        # 3. 카드 검출 → px_per_cm
        # ...

        # 4. 최종 키 산출
        # height_cm = total_px / px_per_cm
        return result
```

**서빙 의존성 (가벼움):**

```txt
# requirements-serve.txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
onnxruntime==1.20.0       # GPU: onnxruntime-gpu
opencv-python-headless==4.10.0
numpy==2.1.0
rtmlib==0.1.0             # RTMPose ONNX 추론
```

학습 때 쓰는 무거운 MMPose/MMDetection은 서빙 서버에 설치하지 않는다. ONNX 모델만 배포.

---

## 7. 클라우드 학습 환경 비교 및 선정 근거

### 7-1. 플랫폼별 스펙 비교

| 항목 | **Colab 무료** | **Colab Pro ($9.99/월)** | **Colab Pro+ ($49.99/월)** | **Kaggle** | **RunPod** |
|------|-------------|----------------------|-------------------------|-----------|-----------|
| GPU | T4 (가끔) | T4 (주로), 가끔 V100 | T4/V100, 가끔 A100 | T4 × 2 (고정) | **직접 선택** (4090, A100 등) |
| 가격 | 무료 | $9.99/월 (100 CU) | $49.99/월 | **무료** (30hr/주) | 초단위 과금 ($0.34~1.74/hr) |
| 세션 시간 | 최대 12시간 | 최대 12시간 | 최대 24시간 | 최대 9시간 | **무제한** |
| 중단 위험 | ⚠️ 높음 (선점, 유휴 감지) | ⚠️ 있음 (CU 소진 시) | ⚠️ 낮지만 있음 | 낮음 (백그라운드 가능) | **없음** |
| 환경 유지 | 세션 끝나면 초기화 | 세션 끝나면 초기화 | 세션 끝나면 초기화 | 세션 끝나면 초기화 | Volume에 영구 유지 |
| 스토리지 | Google Drive 마운트 | Google Drive 마운트 | Google Drive 마운트 | 20GB 영구 | Network Volume ($0.07/GB/월) |

### 7-2. 왜 이렇게 배치했는가

**핵심 기준: "학습이 중간에 끊기면 어떻게 되는가"**

RTMPose WholeBody 파인튜닝은 **24~48시간 연속 학습**이 필요하다. 이 작업에서 Colab이 문제가 되는 이유:

```
Colab Pro로 RTMPose 48시간 학습 시나리오:

시간 0h:   학습 시작 ✅
시간 6h:   "유휴 상태" 감지 → 브라우저 탭 확인 필요 ⚠️
시간 8h:   CU 빠르게 소진 중 ⚠️
시간 11h:  세션 선점 (preemption) → 학습 중단 ❌
           → 체크포인트에서 재개해야 함
           → 환경 재설치 30분
           → 데이터 다시 마운트
시간 12h:  세션 최대 시간 도달 → 강제 종료 ❌

= 12시간 중 실제 학습 ~10시간
= 48시간 학습 완료하려면 5~6번 재시작
= 매번 브라우저 앞에 앉아서 감시해야 함
```

```
RunPod로 같은 작업:

시간 0h:   Pod 생성, 학습 시작 ✅
시간 48h:  학습 완료 ✅ (중간에 아무것도 안 해도 됨)
           → 모델 다운로드 → Pod 종료 → 과금 즉시 중지

= $0.34/hr × 48hr = $16.32 (RTX 4090)
= 자는 동안 학습 완료
```

반면 **카드 검출(RTMDet-nano)은 2~3시간이면 끝나므로** Kaggle 무료 세션(9시간)으로 충분하다. 굳이 RunPod에 돈을 쓸 이유가 없다.

### 7-3. Colab Pro+ ($49.99) vs RunPod 비용 비교

A100이 필요한 경우를 가정하면:

| 항목 | Colab Pro+ | RunPod A100 80GB |
|------|-----------|-----------------|
| 월 구독료 | $49.99 | $0 (사용량만) |
| A100 시간당 CU 소모 | ~13 CU/hr | - |
| 월 100CU로 A100 사용 | **~7.7시간** | - |
| 추가 CU 구매 | $0.10/CU × 13 = **$1.30/hr** | **$1.74/hr** |
| 48시간 학습 총 비용 | $49.99 + (48-7.7)hr × $1.30 = **$102.38** | 48hr × $1.74 = **$83.52** |
| 중단 없이 완료 보장 | ❌ | ✅ |

Colab Pro+가 오히려 비싸고 중단 위험까지 있다. RTX 4090으로 하면 RunPod은 $16.32로 더 저렴해진다.

### 7-4. 작업별 최종 플랫폼 배치

| 작업 | 소요 시간 | 추천 환경 | 이유 | 비용 |
|------|----------|----------|------|------|
| OpenMMLab 설치 테스트 | 30분 | **Colab 무료** | 빠르게 확인만 | $0 |
| 데이터 전처리 / augmentation | 2~3시간 | **Kaggle** | 무료 + 디스크 20GB | $0 |
| RTMDet 카드 검출 학습 | 2~3시간 | **Kaggle** | 무료 T4로 충분 | $0 |
| RTMPose pretrained 추론 테스트 | 1시간 | **Colab 무료** | GPU 잠깐만 필요 | $0 |
| RTMDet 사람 검출 파인튜닝 | 3~6시간 | **Kaggle** | 9시간 세션 안에 완료 | $0 |
| **RTMPose 파인튜닝 (본 학습)** | **24~48시간** | **RunPod RTX 4090** | 중단 없는 연속 학습 필수 | **$8~16** |
| 하이퍼파라미터 탐색 (3~5회) | 각 24시간 | **RunPod RTX 4090** | 반복 학습, 자동화 | **$24~80** |
| ONNX 변환 | 30분 | **Kaggle** | CPU로도 가능 | $0 |

**요약: 무료 환경(Kaggle/Colab)으로 90%의 작업을 처리하고, RunPod은 "포즈 파인튜닝"이라는 단 하나의 무거운 작업에만 집중 투입한다.**

---

## 8. 전체 타임라인 + 비용

```
Week 1: 환경 세팅 + 카드 데이터 수집
├── RunPod/Kaggle에 OpenMMLab 환경 구축
├── Roboflow에서 카드 데이터셋 다운로드
├── 직접 촬영 시작 (카드 + 인형)
├── CVAT 라벨링 환경 세팅
└── 비용: $0

Week 2: 카드 검출 모델 학습 + 사람 검출 테스트
├── RTMDet-nano 카드 검출 학습 (Kaggle T4, 3시간)
├── RTMDet-nano pretrained로 영유아 검출 테스트
├── SyRIP 데이터셋 이메일 요청
├── COCO에서 아이 이미지 필터링 스크립트 작성
└── 비용: $0

Week 3-4: 영유아 데이터 수집 + 라벨링
├── 자체 촬영 데이터 500장 수집
├── CVAT에서 핵심 23kp 라벨링
├── 합성 데이터 5,000장 생성 (RunPod, 3시간)
├── SyRIP 수신 대기
└── 비용: $2~500 (라벨링 자체/외주)

Week 5-6: 포즈 모델 파인튜닝
├── RTMPose-WholeBody-l 영유아 파인튜닝 (RunPod RTX 4090)
├── 사람 검출 파인튜닝 (필요 시)
├── 하이퍼파라미터 탐색 3~5회
├── ONNX 변환
└── 비용: $32~96

Week 7-8: 통합 + 서빙
├── FastAPI + rtmlib로 추론 서버 구축
├── 3개 ONNX 모델 통합 테스트
├── 보정계수 튜닝 (head_top, 영아 머리 비율)
├── Grow Up 앱 연동
└── 비용: $0~20
```

### 비용 요약

| 항목 | 최소 | 최대 |
|------|------|------|
| 카드 검출 학습 (Kaggle) | $0 | $0 |
| 사람 검출 파인튜닝 (Kaggle/RunPod) | $0 | $10 |
| 포즈 모델 파인튜닝 (RunPod RTX 4090) | $32 | $96 |
| 합성 데이터 생성 (RunPod) | $2 | $5 |
| 라벨링 | $0 (자체) | $500 (외주) |
| **합계** | **$34 (~₩48,000)** | **$611 (~₩870,000)** |
| **현실적 예산** | **~$100~200 (₩14만~28만)** | |

---

## 9. 액션 아이템

### 즉시 (이번 주)

- [ ] RunPod 계정 생성 + $25 크레딧 충전
- [ ] Kaggle에서 OpenMMLab 환경 세팅 노트북 작성
- [ ] Roboflow에서 카드 데이터셋 3~5개 다운로드 + COCO JSON 변환
- [ ] SyRIP 데이터셋 접근 요청 이메일 발송
- [ ] CVAT 계정 생성 + 프로젝트 세팅

### 1~2주 내

- [ ] 카드 + 인형 탑뷰 직접 촬영 500장
- [ ] CVAT에서 카드 bbox 라벨링 완료
- [ ] Kaggle에서 RTMDet-nano 카드 검출 첫 학습
- [ ] RTMDet-nano pretrained로 영유아 탑뷰 검출 성능 확인

### 3~4주 내

- [ ] 영유아 촬영 데이터 수집 (보호자 협력)
- [ ] CVAT에서 핵심 23kp 라벨링 시작
- [ ] COCO 2017에서 아이 포함 이미지 자동 필터링
- [ ] 합성 영유아 데이터 생성 파이프라인 구축

### 5~6주 내

- [ ] RTMPose-WholeBody-l 영유아 파인튜닝 (RunPod)
- [ ] MMDeploy로 3개 모델 ONNX 변환
- [ ] rtmlib 기반 FastAPI 추론 서버 프로토타입
- [ ] E2E 테스트 (실제 사진 → 키 측정값 검증)

---

## 10. 라이선스 최종 확인

| 컴포넌트 | 라이선스 | 상용화 |
|---------|---------|-------|
| MMEngine | Apache 2.0 | ✅ |
| MMCV | Apache 2.0 | ✅ |
| MMDetection (RTMDet) | Apache 2.0 | ✅ |
| MMPose (RTMPose) | Apache 2.0 | ✅ |
| MMDeploy | Apache 2.0 | ✅ |
| rtmlib | Apache 2.0 | ✅ |
| COCO-WholeBody | CC BY 4.0 | ✅ (출처 표시) |
| SyRIP | 개별 확인 필요 | ⚠️ 연구자에게 상용 가능 여부 문의 |
| babyPose | CC BY-NC-ND 4.0 | ❌ 벤치마크/참고만 |

**결론: 전체 파이프라인이 Apache 2.0으로 통일. 소스 공개 의무 없이 상용화 가능.**

---

## Version History

| 버전 | 날짜 | 변경 |
|------|------|------|
| v1.0 | 2026-03-09 | 초안 (혼합 프레임워크: YOLO + RTMPose) |
| v1.1 | 2026-03-09 | RTM/OpenMMLab 생태계 통일, config/설치/데이터 가이드, 클라우드 환경 비교 섹션 추가 |
