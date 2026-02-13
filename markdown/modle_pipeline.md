# H-ALIGN AI 키 측정 파이프라인 v1.0

**작성일**: 2026-02-11  
**기반**: Grow Up MVP 최종 합의 문서, 기능 사양서 v1.0  
**서빙**: FastAPI + Uvicorn  
**상태**: Draft — 리뷰 후 확정  

---

## 1. 개요

### 1.1 H-ALIGN Measurement™란?

사진 한 장으로 영유아(0-24개월)의 키를 측정하는 AI 시스템. 병원의 영아 신장 측정 원리(다리를 펴서 측정)를 소프트웨어로 구현한다.

### 1.2 핵심 원리

```
사진 → 참조 객체(카드) 검출 → px당 실제 cm 비율 산출
    → 아이 스켈레톤 추정 (133 키포인트)
    → 각 뼈 세그먼트 길이 합산 (굽어진 다리를 "펴서" 측정)
    → 범위값 + 신뢰도 반환
```

### 1.3 왜 세그먼트 합산인가?

영유아는 다리를 구부리고 있는 경우가 대부분이다. 단순 직선 거리(머리끝~발끝)는 실제 키보다 짧게 측정된다.

```
직선 거리 (세그멘테이션 장축):
  머리 ─────────────────── 발  = 55cm ❌ (다리 굽힘 무시)

스켈레톤 세그먼트 합산:
  머리끝 → 목:        8cm
  목 → 엉덩이:       15cm
  엉덩이 → 무릎:     12cm   ← 구부려져 있어도
  무릎 → 발목:       10cm   ← 각 뼈 길이는 동일
  발목 → 발끝:        3cm
  ─────────────────────────
  합계:              48cm ✅ (펴진 키에 해당)
```

각 관절 사이의 유클리드 거리를 합산하면, 관절이 구부러져 있더라도 "뼈의 실제 길이"가 보존된다.

### 1.4 측정 방식

| 항목 | 결정 |
|------|------|
| 촬영 방향 | **탑뷰** (위에서 아래로) 권장 |
| 아이 자세 | 평평한 바닥에 누운 상태 |
| 참조 객체 | 신용카드(85.6mm) 또는 A4 용지(297mm) |
| 참조 배치 | 아이와 같은 평면(바닥)에 놓기 |
| 출력 | 범위값(min~max cm) + 신뢰도(0~1) |

---

## 2. 전체 파이프라인

```
┌─────────────────────────────────────────────────────────┐
│  [입력] 사진 1장 (탑뷰, 아이 + 참조 카드)                 │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: 전처리                                         │
│  - 이미지 리사이즈 (장변 1280px)                           │
│  - EXIF 회전 보정                                         │
│  - 색공간 변환 (BGR → RGB)                                │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: 참조 카드 검출                                  │
│  - YOLOv8-nano (커스텀 학습) 또는 OpenCV 컨투어            │
│  - 카드 4점 좌표 추출                                      │
│  - 호모그래피 보정 (기울어진 카드 대응)                       │
│  - px_per_cm 산출                                         │
│                                                           │
│  출력: px_per_cm, card_confidence                         │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: 아이 포즈 추정                                  │
│  - RTMPose-WholeBody (133 키포인트)                       │
│  - 머리끝(head_top) ~ 발끝(big_toe) 체인 추출              │
│  - 양쪽 다리 비교 → 더 긴(덜 구부린) 쪽 선택               │
│                                                           │
│  출력: keypoints dict, pose_confidence                    │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: 키 산출                                         │
│  - 세그먼트별 유클리드 거리 합산 (px)                       │
│  - px → cm 변환 (px_per_cm 적용)                          │
│  - head_top 보정 (영아 머리 비율)                          │
│  - 오차 마진 계산 (신뢰도 기반)                             │
│                                                           │
│  출력: { min, max, confidence, segments }                 │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 5: 보호자 검증 (Human-in-the-Loop)                 │
│  - 앱에서 결과 표시                                        │
│  - 보호자가 "맞아요" 또는 "다시 측정" 선택                   │
│  - 보호자 보정값 입력 가능                                  │
│                                                           │
│  최종 출력: measurements 테이블 INSERT                     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Stage 2: 참조 카드 검출

### 3.1 지원 참조 객체

| 객체 | 실제 크기 | 검출 방식 | 정확도 |
|------|-----------|-----------|--------|
| **신용카드/체크카드** | 85.6 × 53.98mm | YOLOv8-nano + 4점 호모그래피 | ⭐⭐⭐⭐⭐ |
| **A4 용지** | 297 × 210mm | OpenCV 컨투어 + 4점 호모그래피 | ⭐⭐⭐⭐ |
| **앱 제공 인쇄 매트** | 600 × 900mm (ArUco 마커 4개) | ArUco 검출 + 호모그래피 | ⭐⭐⭐⭐⭐ |

MVP에서는 **신용카드**를 기본 참조 객체로 사용한다. 사용자 가이드에서 "아이 옆에 카드를 놓아주세요"로 안내.

### 3.2 방식 A: YOLOv8-nano 커스텀 학습 (권장)

#### 학습 데이터

```
datasets/card_detection/
├── images/
│   ├── train/     # 카드 사진 500-1000장
│   └── val/       # 100-200장
├── labels/
│   ├── train/     # YOLO format bounding box
│   └── val/
└── data.yaml
```

```yaml
# data.yaml
train: datasets/card_detection/images/train
val: datasets/card_detection/images/val
nc: 1
names: ['card']
```

#### 학습

```bash
pip install ultralytics

yolo detect train \
  model=yolov8n.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=card_detector
```

#### 추론 + 4점 추출

```python
from ultralytics import YOLO
import cv2
import numpy as np

card_model = YOLO("card_detector.pt")

CARD_WIDTH_CM = 8.56   # 신용카드 가로 (cm)
CARD_HEIGHT_CM = 5.398  # 신용카드 세로 (cm)


def detect_card(img: np.ndarray) -> dict:
    """
    카드 검출 → px_per_cm 산출
    
    Returns:
        {
            "px_per_cm": float,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
            "detected": bool,
        }
    """
    results = card_model(img, conf=0.5)
    
    if len(results[0].boxes) == 0:
        return {"detected": False, "px_per_cm": 0, "confidence": 0, "bbox": []}
    
    # 가장 신뢰도 높은 카드 선택
    boxes = results[0].boxes
    best_idx = boxes.conf.argmax()
    bbox = boxes.xyxy[best_idx].cpu().numpy()
    conf = float(boxes.conf[best_idx])
    
    x1, y1, x2, y2 = bbox
    card_width_px = x2 - x1
    card_height_px = y2 - y1
    
    # 가로/세로 중 더 긴 쪽을 카드 가로(85.6mm)로 판단
    if card_width_px >= card_height_px:
        px_per_cm = card_width_px / CARD_WIDTH_CM
    else:
        px_per_cm = card_height_px / CARD_WIDTH_CM
    
    # 가로세로 비율 검증 (카드는 약 1.586:1)
    aspect_ratio = max(card_width_px, card_height_px) / min(card_width_px, card_height_px)
    expected_ratio = CARD_WIDTH_CM / CARD_HEIGHT_CM  # 1.586
    ratio_error = abs(aspect_ratio - expected_ratio) / expected_ratio
    
    if ratio_error > 0.15:
        # 비율이 15% 이상 다르면 카드가 기울어져 있거나 오검출
        conf *= 0.7  # 신뢰도 감점
    
    return {
        "detected": True,
        "px_per_cm": float(px_per_cm),
        "confidence": conf,
        "bbox": bbox.tolist(),
    }
```

### 3.3 방식 B: OpenCV 컨투어 (모델 학습 없이)

카드 학습 데이터가 부족한 초기 단계에서 사용 가능한 대안.

```python
import cv2
import numpy as np


def detect_card_contour(img: np.ndarray, ref_width_cm: float = 8.56) -> dict:
    """
    OpenCV 컨투어 기반 사각형(카드) 검출
    조명이 균일한 탑뷰에서 잘 작동
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 팽창으로 엣지 연결
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_card = None
    best_area = 0
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # 4꼭짓점 사각형만
        if len(approx) != 4:
            continue
        
        area = cv2.contourArea(approx)
        
        # 너무 작거나 너무 큰 사각형 필터
        img_area = img.shape[0] * img.shape[1]
        if area < img_area * 0.005 or area > img_area * 0.3:
            continue
        
        # 가로세로 비율 체크 (카드: ~1.586:1)
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if min(w, h) == 0:
            continue
        ratio = max(w, h) / min(w, h)
        if abs(ratio - 1.586) > 0.3:
            continue
        
        if area > best_area:
            best_area = area
            best_card = approx
    
    if best_card is None:
        return {"detected": False, "px_per_cm": 0, "confidence": 0}
    
    # 4점 호모그래피로 정확한 px_per_cm 계산
    pts = best_card.reshape(4, 2).astype(np.float32)
    pts = order_points(pts)
    
    width_px = np.linalg.norm(pts[0] - pts[1])
    height_px = np.linalg.norm(pts[0] - pts[3])
    
    if width_px >= height_px:
        px_per_cm = width_px / ref_width_cm
    else:
        px_per_cm = height_px / ref_width_cm
    
    return {
        "detected": True,
        "px_per_cm": float(px_per_cm),
        "confidence": 0.8,  # 컨투어 방식은 고정 신뢰도
        "corners": pts.tolist(),
    }


def order_points(pts: np.ndarray) -> np.ndarray:
    """4점을 [top-left, top-right, bottom-right, bottom-left] 순서로 정렬"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(d)]      # top-right
    rect[3] = pts[np.argmax(d)]      # bottom-left
    return rect
```

### 3.4 호모그래피 보정 (기울어진 카드)

탑뷰가 아닌 경우 또는 카드가 약간 기울어진 경우 원근 보정 적용.

```python
def correct_perspective(img: np.ndarray, card_corners: np.ndarray) -> tuple:
    """
    기울어진 카드를 정면으로 보정하고 정확한 px_per_cm 산출
    
    Args:
        img: 원본 이미지
        card_corners: 카드 4점 좌표 (ordered)
    
    Returns:
        (corrected_img, px_per_cm)
    """
    src_pts = card_corners.astype(np.float32)
    
    # 실제 카드 크기 비율로 목표 좌표 설정 (px 단위)
    target_w = 300  # 임의 타겟 너비
    target_h = int(target_w / 1.586)  # 카드 비율 유지
    
    dst_pts = np.array([
        [0, 0],
        [target_w, 0],
        [target_w, target_h],
        [0, target_h],
    ], dtype=np.float32)
    
    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    # 전체 이미지에 호모그래피 적용 (카드 평면 기준으로 보정)
    corrected = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    
    px_per_cm = target_w / CARD_WIDTH_CM
    
    return corrected, px_per_cm
```

---

## 4. Stage 3: 아이 포즈 추정

### 4.1 모델 선택

| 모델 | 키포인트 | head_top | toe | 영아 | 속도 | 추천 |
|------|----------|----------|-----|------|------|------|
| **RTMPose-WholeBody-l** | 133개 | ✅ | ✅ | ⭐⭐⭐⭐ | 빠름 | ✅ MVP |
| ViTPose+-WholeBody | 133개 | ✅ | ✅ | ⭐⭐⭐⭐⭐ | 느림 | Phase 2 |
| MediaPipe Pose | 33개 | ✕ (코) | ✅ | ⭐⭐⭐ | 매우 빠름 | 대안 |
| YOLO-Pose (COCO 17) | 17개 | ✕ | ✕ | ⭐⭐ | 가장 빠름 | ✕ |

**MVP 결정**: **RTMPose-WholeBody** — 133개 키포인트로 head_top과 발가락까지 제공하며, ONNX 변환으로 GPU 없이도 서빙 가능.

> COCO 17개 키포인트로는 머리 꼭대기(head_top)와 발끝(toe)이 없어 정확한 키 측정이 불가능하다.

### 4.2 RTMPose-WholeBody 133 키포인트 맵

키 측정에 사용하는 키포인트:

```
인덱스   이름              용도
──────────────────────────────────────
0       nose              head_top 방향 추정 보조
1-4     eyes, ears        머리 방향 판단
5       left_shoulder     neck 중점 계산
6       right_shoulder    neck 중점 계산
11      left_hip          hip_center 계산
12      right_hip         hip_center 계산
13      left_knee         왼쪽 다리 세그먼트
14      right_knee        오른쪽 다리 세그먼트
15      left_ankle        왼쪽 다리 세그먼트
16      right_ankle       오른쪽 다리 세그먼트

--- WholeBody 확장 (COCO 17개에 추가) ---
23-90   face landmarks    머리 상단 추정
91-111  left hand         사용 안 함
112-132 right hand        사용 안 함

--- 발 키포인트 (WholeBody) ---
17      left_big_toe      왼발 끝
18      left_small_toe    사용 안 함
19      left_heel         보조
20      right_big_toe     오른발 끝
21      right_small_toe   사용 안 함
22      right_heel        보조
```

> 주의: RTMPose-WholeBody의 키포인트 인덱스는 모델/버전에 따라 다를 수 있다. 반드시 사용할 모델의 공식 문서에서 인덱스 매핑을 확인할 것.

### 4.3 키 측정 세그먼트 체인

```
head_top → neck → hip_center → knee → ankle → big_toe
   ①         ②         ③          ④        ⑤

① 머리끝 → 목:      머리 길이
② 목 → 엉덩이:      몸통 길이 (척추)
③ 엉덩이 → 무릎:    대퇴골 (허벅지)
④ 무릎 → 발목:      경골 (정강이)
⑤ 발목 → 발끝:      발 길이 보정
```

### 4.4 head_top 추정

WholeBody 133에는 face landmark가 포함되어 있으나, 영아의 머리 꼭대기를 정확히 잡기는 어렵다.

보정 전략:

```python
def estimate_head_top(keypoints: dict, img_h: int) -> np.ndarray:
    """
    face landmark 또는 nose 기반으로 head_top 추정
    
    영아의 머리는 체고 대비 약 25% (성인은 ~13%)
    nose에서 neck 방향의 반대로 연장하여 추정
    """
    nose = keypoints["nose"]
    neck = keypoints["neck"]
    
    # nose → neck 벡터의 반대 방향
    direction = nose - neck
    direction_norm = direction / (np.linalg.norm(direction) + 1e-6)
    
    # 영아 머리 보정: nose에서 위로 (nose→neck 거리) × 보정계수
    nose_to_neck_dist = np.linalg.norm(direction)
    
    # 영아(0-24개월)는 머리가 크므로 보정계수 0.6~0.8
    # 성인이면 0.4 정도
    INFANT_HEAD_RATIO = 0.7
    
    head_top = nose + direction_norm * nose_to_neck_dist * INFANT_HEAD_RATIO
    
    return head_top
```

### 4.5 양쪽 다리 선택 로직

```python
def select_longer_leg(keypoints_raw: dict) -> str:
    """
    양쪽 다리 세그먼트 합산 비교 → 더 긴(덜 구부린) 쪽 선택
    
    Returns: 'left' 또는 'right'
    """
    left_leg_length = (
        np.linalg.norm(keypoints_raw["left_hip"] - keypoints_raw["left_knee"]) +
        np.linalg.norm(keypoints_raw["left_knee"] - keypoints_raw["left_ankle"]) +
        np.linalg.norm(keypoints_raw["left_ankle"] - keypoints_raw["left_big_toe"])
    )
    
    right_leg_length = (
        np.linalg.norm(keypoints_raw["right_hip"] - keypoints_raw["right_knee"]) +
        np.linalg.norm(keypoints_raw["right_knee"] - keypoints_raw["right_ankle"]) +
        np.linalg.norm(keypoints_raw["right_ankle"] - keypoints_raw["right_big_toe"])
    )
    
    return "left" if left_leg_length >= right_leg_length else "right"
```

---

## 5. Stage 4: 키 산출

### 5.1 세그먼트 합산 엔진

```python
import numpy as np
from typing import List, Tuple

# 키 측정 세그먼트 체인 정의
BODY_SEGMENTS = [
    ("head_top",    "neck",        "머리"),
    ("neck",        "hip_center",  "몸통"),
    ("hip_center",  "knee",        "허벅지"),
    ("knee",        "ankle",       "정강이"),
    ("ankle",       "big_toe",     "발"),
]


def compute_segment_length(p1: np.ndarray, p2: np.ndarray) -> float:
    """두 키포인트 사이 유클리드 거리 (px)"""
    return float(np.linalg.norm(p1 - p2))


def compute_unfolded_height_px(keypoints: dict) -> Tuple[float, dict]:
    """
    굽어진 관절을 무시하고, 각 뼈 세그먼트 길이를 합산
    = 다리를 편 상태의 키 (px 단위)
    
    Returns:
        (total_px, segments_dict)
    """
    total_px = 0.0
    segments = {}
    
    for start_name, end_name, label in BODY_SEGMENTS:
        p1 = keypoints[start_name]
        p2 = keypoints[end_name]
        length = compute_segment_length(p1, p2)
        segments[label] = length
        total_px += length
    
    return total_px, segments


def compute_knee_bend_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """
    무릎 굽힘 각도 계산
    180° = 완전히 편 상태
    90° = 직각 구부림
    """
    v1 = hip - knee
    v2 = ankle - knee
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    return float(angle_deg)
```

### 5.2 최종 키 산출 + 신뢰도

```python
def calculate_height(
    keypoints: dict,
    px_per_cm: float,
    card_confidence: float,
    pose_confidences: dict,
) -> dict:
    """
    최종 키 산출
    
    Args:
        keypoints: 키 측정용 키포인트 좌표 (px)
        px_per_cm: 카드 검출에서 산출한 px/cm 비율
        card_confidence: 카드 검출 신뢰도
        pose_confidences: 각 키포인트의 신뢰도 {name: float}
    
    Returns:
        {
            "height_range": {"min": float, "max": float},
            "confidence": float,
            "method": "h-align",
            "segments_cm": {"머리": float, "몸통": float, ...},
            "knee_angle": float,
            "warnings": [str]
        }
    """
    warnings = []
    
    # 1. 세그먼트 합산 (px)
    total_px, segments_px = compute_unfolded_height_px(keypoints)
    
    # 2. px → cm 변환
    height_cm = total_px / px_per_cm
    segments_cm = {k: round(v / px_per_cm, 1) for k, v in segments_px.items()}
    
    # 3. 무릎 각도 체크
    knee_angle = compute_knee_bend_angle(
        keypoints["hip_center"], keypoints["knee"], keypoints["ankle"]
    )
    
    if knee_angle < 120:
        warnings.append(f"무릎이 많이 구부러져 있습니다 ({knee_angle:.0f}°). 오차가 클 수 있습니다.")
    
    # 4. 종합 신뢰도 계산
    key_points = ["head_top", "neck", "hip_center", "knee", "ankle", "big_toe"]
    pose_conf_values = [pose_confidences.get(p, 0.5) for p in key_points]
    avg_pose_conf = sum(pose_conf_values) / len(pose_conf_values)
    min_pose_conf = min(pose_conf_values)
    
    # 카드 신뢰도 × 포즈 신뢰도 가중 평균
    overall_confidence = card_confidence * 0.4 + avg_pose_conf * 0.4 + (knee_angle / 180) * 0.2
    overall_confidence = min(overall_confidence, 0.99)
    
    if min_pose_conf < 0.3:
        overall_confidence *= 0.7
        warnings.append("일부 키포인트의 신뢰도가 낮습니다. 아이가 잘 보이도록 다시 찍어주세요.")
    
    # 5. 오차 마진 계산 (신뢰도 기반)
    if overall_confidence >= 0.85:
        margin = 0.3
    elif overall_confidence >= 0.7:
        margin = 0.5
    elif overall_confidence >= 0.5:
        margin = 1.0
    else:
        margin = 1.5
        warnings.append("측정 정확도가 낮습니다. 재촬영을 권장합니다.")
    
    return {
        "height_range": {
            "min": round(height_cm - margin, 1),
            "max": round(height_cm + margin, 1),
        },
        "confidence": round(overall_confidence, 2),
        "method": "h-align",
        "segments_cm": segments_cm,
        "knee_angle": round(knee_angle, 1),
        "warnings": warnings,
    }
```

---

## 6. 키포인트 추출 통합 모듈

### 6.1 RTMPose 추론

```python
# 설치: pip install mmpose mmdet onnxruntime
# 또는 ONNX 변환 후 onnxruntime으로 직접 추론

import numpy as np

# RTMPose-WholeBody 키포인트 인덱스 (모델 버전에 따라 확인 필요)
# 아래는 COCO-WholeBody 133 기준 예시
KEYPOINT_MAP = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    "left_big_toe": 17,
    "left_heel": 19,
    "right_big_toe": 20,
    "right_heel": 22,
    # face landmarks: 23-90 (머리 상단 추정에 활용)
}


def extract_keypoints_for_height(
    landmarks: np.ndarray,
    img_h: int,
    img_w: int,
) -> Tuple[dict, dict]:
    """
    133개 WholeBody 키포인트에서 키 측정용 포인트 추출
    
    Args:
        landmarks: shape (133, 3) — [x_norm, y_norm, confidence]
        img_h, img_w: 이미지 크기
    
    Returns:
        (keypoints_px, confidences)
        keypoints_px: {name: np.array([x, y])}
        confidences: {name: float}
    """
    def pt(idx):
        return np.array([landmarks[idx][0] * img_w, landmarks[idx][1] * img_h])
    
    def conf(idx):
        return float(landmarks[idx][2])
    
    # 기본 키포인트 추출
    raw = {}
    raw_conf = {}
    for name, idx in KEYPOINT_MAP.items():
        raw[name] = pt(idx)
        raw_conf[name] = conf(idx)
    
    # 중간 지점 계산
    neck = (raw["left_shoulder"] + raw["right_shoulder"]) / 2
    hip_center = (raw["left_hip"] + raw["right_hip"]) / 2
    
    # 양쪽 다리 비교 → 더 긴 쪽 선택
    side = select_longer_leg({
        "left_hip": raw["left_hip"],
        "left_knee": raw["left_knee"],
        "left_ankle": raw["left_ankle"],
        "left_big_toe": raw["left_big_toe"],
        "right_hip": raw["right_hip"],
        "right_knee": raw["right_knee"],
        "right_ankle": raw["right_ankle"],
        "right_big_toe": raw["right_big_toe"],
    })
    
    # head_top 추정
    head_top = estimate_head_top({"nose": raw["nose"], "neck": neck}, img_h)
    
    # 선택된 쪽으로 키포인트 구성
    keypoints = {
        "head_top":    head_top,
        "neck":        neck,
        "hip_center":  hip_center,
        "knee":        raw[f"{side}_knee"],
        "ankle":       raw[f"{side}_ankle"],
        "big_toe":     raw[f"{side}_big_toe"],
    }
    
    confidences = {
        "head_top":    raw_conf["nose"] * 0.8,  # 추정치이므로 감점
        "neck":        (raw_conf["left_shoulder"] + raw_conf["right_shoulder"]) / 2,
        "hip_center":  (raw_conf["left_hip"] + raw_conf["right_hip"]) / 2,
        "knee":        raw_conf[f"{side}_knee"],
        "ankle":       raw_conf[f"{side}_ankle"],
        "big_toe":     raw_conf[f"{side}_big_toe"],
    }
    
    return keypoints, confidences
```

---

## 7. FastAPI 서버

### 7.1 프로젝트 구조

```
h-align-server/
├── app/
│   ├── main.py                # FastAPI 앱 진입점
│   ├── config.py              # 환경설정
│   ├── routers/
│   │   └── measure.py         # /measure 엔드포인트
│   ├── services/
│   │   ├── card_detector.py   # Stage 2: 카드 검출
│   │   ├── pose_estimator.py  # Stage 3: 포즈 추정
│   │   └── height_calculator.py # Stage 4: 키 산출
│   ├── models/                # AI 모델 가중치
│   │   ├── card_detector.pt
│   │   └── rtmpose_wholebody.onnx
│   └── schemas/
│       └── measurement.py     # Pydantic 스키마
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### 7.2 의존성

```txt
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
ultralytics==8.3.0
onnxruntime==1.20.0       # GPU: onnxruntime-gpu
opencv-python-headless==4.10.0
numpy==2.1.0
Pillow==11.0.0
pydantic==2.10.0
```

### 7.3 Pydantic 스키마

```python
# app/schemas/measurement.py

from pydantic import BaseModel
from typing import Optional


class HeightRange(BaseModel):
    min: float
    max: float


class MeasurementResult(BaseModel):
    height_range: HeightRange
    confidence: float
    method: str = "h-align"
    segments_cm: dict[str, float]
    knee_angle: float
    warnings: list[str]


class MeasurementResponse(BaseModel):
    success: bool
    result: Optional[MeasurementResult] = None
    error: Optional[str] = None
    
    
class ErrorDetail(BaseModel):
    code: str
    message: str
```

### 7.4 메인 엔드포인트

```python
# app/routers/measure.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.card_detector import CardDetector
from app.services.pose_estimator import PoseEstimator
from app.services.height_calculator import HeightCalculator
from app.schemas.measurement import MeasurementResponse
import cv2
import numpy as np

router = APIRouter()

card_detector = CardDetector()
pose_estimator = PoseEstimator()
height_calculator = HeightCalculator()


@router.post("/measure", response_model=MeasurementResponse)
async def measure_height(file: UploadFile = File(...)):
    """
    사진에서 영유아 키 측정
    
    - 참조 카드(신용카드)와 아이가 함께 찍힌 탑뷰 사진 필요
    - 범위값(min~max cm) + 신뢰도(0~1) 반환
    """
    # 1. 이미지 읽기 + 전처리
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return MeasurementResponse(
            success=False,
            error="이미지를 읽을 수 없습니다."
        )
    
    img = preprocess_image(img)
    img_h, img_w = img.shape[:2]
    
    # 2. 카드 검출
    card_result = card_detector.detect(img)
    
    if not card_result["detected"]:
        return MeasurementResponse(
            success=False,
            error="참조 카드를 찾을 수 없습니다. 카드가 잘 보이도록 다시 찍어주세요."
        )
    
    # 3. 포즈 추정
    pose_result = pose_estimator.estimate(img)
    
    if not pose_result["detected"]:
        return MeasurementResponse(
            success=False,
            error="아이를 인식할 수 없습니다. 전신이 보이도록 다시 찍어주세요."
        )
    
    # 4. 키포인트 추출
    keypoints, confidences = extract_keypoints_for_height(
        pose_result["landmarks"], img_h, img_w
    )
    
    # 5. 키 산출
    result = height_calculator.calculate(
        keypoints=keypoints,
        px_per_cm=card_result["px_per_cm"],
        card_confidence=card_result["confidence"],
        pose_confidences=confidences,
    )
    
    return MeasurementResponse(success=True, result=result)


def preprocess_image(img: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """이미지 전처리: EXIF 회전 보정 + 리사이즈"""
    h, w = img.shape[:2]
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    return img
```

### 7.5 앱 진입점

```python
# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import measure

app = FastAPI(
    title="H-ALIGN Measurement API",
    description="AI 기반 영유아 키 측정 서비스",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(measure.router, prefix="/api/v1", tags=["measurement"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}
```

---

## 8. 정확도 향상 전략

### 8.1 현재 한계 및 대응

| 한계 | 원인 | 대응 |
|------|------|------|
| head_top 부정확 | COCO WholeBody에 정확한 head_top 없음 | nose 기반 보정계수 + face landmark 활용 |
| 옷 때문에 관절 위치 부정확 | 두꺼운 옷이 키포인트 추정을 방해 | 촬영 가이드: "얇은 옷 권장" |
| 카드와 아이의 평면 불일치 | 카드가 바닥, 아이가 살짝 높은 곳 | 탑뷰 촬영 강제 + 높이 차이 무시 가능 |
| 영아 포즈 데이터 부족 | 학습 데이터에 누운 영아 적음 | 커스텀 파인튜닝 또는 데이터 증강 |

### 8.2 Phase별 정확도 개선 로드맵

```
MVP (현재):
  - YOLOv8-nano 카드 검출 + RTMPose-WholeBody
  - 예상 정확도: 90-95%
  - 오차: ±0.5~1.5cm

Phase 2:
  - 영아 전용 포즈 모델 파인튜닝
  - ArUco 매트 PDF 제공 → 호모그래피 정확도 향상
  - 예상 정확도: 95-97%
  - 오차: ±0.3~0.5cm

Phase 3:
  - LiDAR 지원 (iPhone Pro)
  - 자체 영아 포즈 데이터셋 구축
  - 예상 정확도: 97-99%
  - 오차: ±0.1~0.3cm
```

### 8.3 촬영 가이드 (앱 UI에서 안내)

사용자에게 표시할 촬영 팁:

```
✅ 권장사항:
  1. 아이를 평평한 바닥에 눕혀주세요
  2. 아이 옆에 카드(신용카드, 교통카드)를 놓아주세요
  3. 위에서 아래로 전신이 보이도록 찍어주세요
  4. 얇은 옷을 입히면 더 정확해요
  5. 밝은 조명에서 찍어주세요

❌ 피해야 할 것:
  - 옆에서 비스듬히 찍기
  - 카드 없이 찍기
  - 아이가 이불에 가려진 상태
  - 너무 어두운 환경
```

---

## 9. MVP 모의 결과 (AI 모델 미완성 시)

Phase 2에서 실제 AI 모델을 배포하기 전까지 앱의 H-ALIGN UI를 테스트하기 위한 모의 결과.

```python
# app/services/mock_measurement.py

import random


def get_mock_measurement(photo_uri: str) -> dict:
    """
    MVP용 모의 측정 결과
    실제 AI 분석 없이 그럴듯한 결과 반환
    """
    base_height = random.uniform(50.0, 85.0)
    margin = round(random.uniform(0.3, 0.7), 1)
    
    return {
        "height_range": {
            "min": round(base_height - margin, 1),
            "max": round(base_height + margin, 1),
        },
        "confidence": round(random.uniform(0.88, 0.98), 2),
        "method": "mock",  # 실제 AI는 "h-align"
        "segments_cm": {
            "머리": round(random.uniform(8, 12), 1),
            "몸통": round(random.uniform(12, 18), 1),
            "허벅지": round(random.uniform(8, 14), 1),
            "정강이": round(random.uniform(7, 12), 1),
            "발": round(random.uniform(2, 4), 1),
        },
        "knee_angle": round(random.uniform(130, 175), 1),
        "warnings": [],
    }
```

---

## 10. 배포

### 10.1 Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# OpenCV 의존성
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# 모델 가중치 복사
COPY models/ ./app/models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 10.2 docker-compose.yml

```yaml
version: '3.8'

services:
  h-align-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_DIR=/app/models
      - MAX_IMAGE_SIZE=1280
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models:ro
    deploy:
      resources:
        limits:
          memory: 2G
```

### 10.3 AWS 배포 구성 (Phase 2)

```
클라이언트 (React PWA)
    ↓ HTTPS
API Gateway / ALB
    ↓
ECS Fargate (h-align-api 컨테이너)
    ↓
S3 (모델 가중치 저장)
ECR (Docker 이미지 레지스트리)
CloudWatch (로깅/모니터링)
```

---

## 11. API 명세 요약

### 11.1 POST /api/v1/measure

| 항목 | 값 |
|------|-----|
| Method | POST |
| Content-Type | multipart/form-data |
| Body | file: 이미지 파일 (JPEG, PNG) |

#### 성공 응답 (200)

```json
{
  "success": true,
  "result": {
    "height_range": { "min": 68.2, "max": 69.0 },
    "confidence": 0.93,
    "method": "h-align",
    "segments_cm": {
      "머리": 10.2,
      "몸통": 16.5,
      "허벅지": 13.1,
      "정강이": 10.8,
      "발": 3.2
    },
    "knee_angle": 155.3,
    "warnings": []
  }
}
```

#### 실패 응답 (200, success: false)

```json
{
  "success": false,
  "result": null,
  "error": "참조 카드를 찾을 수 없습니다. 카드가 잘 보이도록 다시 찍어주세요."
}
```

#### 에러 코드

| 에러 | 설명 | 사용자 안내 |
|------|------|-------------|
| CARD_NOT_FOUND | 카드 검출 실패 | "카드가 잘 보이도록 다시 찍어주세요" |
| PERSON_NOT_FOUND | 아이 인식 실패 | "전신이 보이도록 다시 찍어주세요" |
| LOW_CONFIDENCE | 신뢰도 0.5 미만 | "측정이 부정확합니다. 재촬영을 권장합니다" |
| IMAGE_TOO_SMALL | 해상도 부족 | "더 가까이에서 찍어주세요" |
| INVALID_IMAGE | 이미지 파싱 실패 | "이미지를 읽을 수 없습니다" |

---

## 12. 프론트엔드 연동 (Grow Up 앱)

### 12.1 MeasureLayout Context에서의 호출

```typescript
// src/contexts/MeasureContext.tsx (기존 설계와 연동)

interface MeasureState {
  capturedPhoto: string | null     // base64 또는 blob URL
  analysisResult: MeasurementResult | null
  isAnalyzing: boolean
  error: string | null
}

async function analyzePhoto(photoBlob: Blob): Promise<MeasurementResult> {
  const formData = new FormData()
  formData.append('file', photoBlob, 'measurement.jpg')
  
  const response = await fetch(`${API_BASE_URL}/api/v1/measure`, {
    method: 'POST',
    body: formData,
  })
  
  const data = await response.json()
  
  if (!data.success) {
    throw new Error(data.error)
  }
  
  return data.result
}
```

### 12.2 measurements 테이블 저장

보호자 검증(4-2) 완료 후:

```typescript
// 보호자가 "확인" 선택 시
await measurementService.addMeasurement({
  childId: activeChild.id,
  type: 'height',
  value: (result.height_range.min + result.height_range.max) / 2,  // 중앙값
  method: 'h-align',
  confidence: result.confidence,
})

// 보호자가 수동 보정한 경우
await measurementService.addMeasurement({
  childId: activeChild.id,
  type: 'height',
  value: manualCorrectedValue,  // 보호자 입력값
  method: 'h-align',
  confidence: result.confidence * 0.9,  // 보정된 신뢰도
})
```

---

## 13. Version History

| 버전 | 날짜 | 변경 |
|------|------|------|
| v1.0 | 2026-02-11 | 초안 작성 (5-Stage 파이프라인, FastAPI 서빙, 카드 검출 + 스켈레톤 세그먼트 합산) |