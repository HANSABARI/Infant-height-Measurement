# 📏 h-align-server (nana-AI)

------------------------------------------------------------------------------------------------------------------------
An AI-powered FastAPI backend server that calculates a person's actual height from an image. 
The pipeline relies on detecting a standard-sized reference object (a credit card) and extracting human body keypoints.

## 🌟 Core AI Pipeline
1. **Card Segmentation (HaS Image Model FP32):** Detects sensitive-document classes, but uses only standard `id_card` and `bank_card` masks to calculate the `pixel-per-cm` ratio from their 8.56cm long side. Passports and employee badges are not valid scale references.
2. **Pose Estimation:** Keeps 13 existing WholeBody keypoints from ONNX Runtime (`rtmlib`) and adds only `head_top` from a separately trained one-keypoint RTMPose model.
3. **Height Calculation:** Combines the skeleton data and the pixel ratio to estimate the actual physical height.

## 📂 Project Structure

```text
h-align-server/
├── requirements.txt              # Core Python dependencies
├── app/                          # Main Application Directory
│   ├── main.py                   # FastAPI application entry point
│   ├── routers/                  # API endpoints (/measure, /measure/debug)
│   ├── schemas/                  # Pydantic models for request/response
│   ├── services/                 # AI & Business Logic
│   │   ├── card_detector_has.py  # HaS Image Model FP32 segmentation inference
│   │   ├── card_geometry.py      # Mask geometry and pixel-per-cm helpers
│   │   ├── card_detector_rtm.py  # Previous RTMDet Inference path
│   │   ├── pose_estimator.py     # Existing WholeBody keypoints
│   │   ├── head_top_pose_estimator.py # WholeBody + head_top fusion
│   │   ├── height_calculator.py  # Physical height calculation logic
│   │   └── visualizer.py         # Drawing bounding boxes and skeletons
│   └── models/                   # Pre-trained weights and config files
│       └── has_image_0209_fp32/
│           └── sensitive_seg_best.pt # Git ignored; auto-downloadable from Hugging Face
├── dataset/                      # (Git Ignored) Training image datasets
└── debug_images/                 # (Git Ignored) Saved debug visualization images
```

⚙️ Installation Guide (macOS Apple Silicon / MPS)
⚠️ CRITICAL NOTE FOR MAC USERS (M1/M2/M3):
Do NOT install mmcv via requirements.txt. To prevent C++ custom operation (e.g., NMS) 
compilation errors on Apple Silicon, you must install the OpenMMLab libraries exactly as outlined in Step 2.

Step 1. Install Basic Requirements
Make sure you have Python 3.10.x installed. Create a virtual environment and install the core dependencies:

```Bash
pip install -r requirements.txt
```

Step 2. Install OpenMMLab Core Libraries
Execute the following commands in your terminal in this exact order to install the exact versions compatible with this project:


# 1. Install MMCV with operations disabled (Crucial for Apple Silicon)

```Bash
MMCV_WITH_OPS=0 mim install "mmcv==2.1.0"
```

# 2. Install MMDetection and MMPose
```Bash
mim install "mmdet==3.2.0"
mim install "mmengine==0.10.7"
mim install "mmpose==1.3.2"
# mmpose chumpy error:
pip install chumpy --no-build-isolation
```

### HaS Image Model FP32 card segmentation weights

The server uses `xuanwulab/HaS_Image_0209_FP32` (`HaS Image Model (FP32)`) by default. Its architecture is YOLO11 instance segmentation.

- Default local path: `app/models/has_image_0209_fp32/sensitive_seg_best.pt`
- Override path: `CARD_HAS_MODEL=/path/to/sensitive_seg_best.pt`
- Disable startup auto-download: `CARD_HAS_AUTO_DOWNLOAD=0`
- Optional device override: `CARD_HAS_DEVICE=cuda:0`
- Optional inference size override: `CARD_HAS_IMGSZ=1920`

🚨 Known Issues & Workarounds
1. PyTorch 2.6+ Security Policy (torch.load UnpicklingError)
Starting from PyTorch 2.6, the default behavior of torch.load has been restricted (weights_only=True by default) for security reasons.
Loading OpenMMLab weights (.pth) will trigger an UnpicklingError due to embedded numpy objects and history buffers.

Solution Applied: The previous RTMDet path keeps its monkey-patching workaround inside app/services/card_detector_rtm.py.
The active HaS Image path uses Ultralytics and does not require the RTMDet config/checkpoint pair.


2. Training on Mac (MPS NMS Limitation)
If you wish to retrain the RTMDet model on an Apple Silicon Mac, note that PyTorch's MPS backend currently does not support the nms (Non-Maximum Suppression) operation.
Solution Applied: In the training config (rtmdet_nano_card.py), validation (val_cfg) and testing (test_cfg) are explicitly disabled (None) to prevent crashes during the training loop.

🚀 Usage
Start the FastAPI server locally:

```Bash
uvicorn app.main:app --reload
```

The server will be available at http://127.0.0.1:8000.

API Documentation (Swagger UI): http://127.0.0.1:8000/docs

Main Endpoint: `POST /api/v1/measure`

Debug Endpoint (disabled by default; saves an image locally): `POST /api/v1/measure/debug`

### jaram-height-web Worker connection

The main endpoint follows the `jaram-height-web` AI contract. It accepts a
normalized image in `file`, `X-Measurement-Id`, and a Bearer token. It returns
only the estimated height, range, confidence, quality, warnings, and model
version; internal card and pose diagnostics are not returned.

Set a shared key before starting the server:

```bash
export H_ALIGN_AI_API_KEY='<shared-ai-api-key>'
uvicorn app.main:app --reload
```

Configure the same value as `AI_API_KEY` in the `jaram-height-web` Edge
Function secrets. The detailed contract is in
[`docs/H_ALIGN_API_SPEC.md`](docs/H_ALIGN_API_SPEC.md).

The debug endpoint is intended for trusted local development only. It returns
`404` unless it is explicitly enabled, and it always requires the same Bearer
token as the main endpoint:

```bash
export H_ALIGN_DEBUG_API_ENABLED=1
export H_ALIGN_AI_API_KEY='<shared-ai-api-key>'
```

Do not enable the debug endpoint on an internet-exposed production server
unless its diagnostic output and locally saved images are explicitly required.

### macOS Docker CPU integration test

Apple Silicon Mac에서는 Linux `amd64` AI 컨테이너를 CPU 모드로 실행해
`jaram-height-web`의 로컬 Supabase Worker와 실제 모델을 통합 검증할 수 있습니다.
이 구성은 API와 비동기 처리 흐름을 확인하기 위한 것이며 Windows CUDA 운영
성능의 기준이 아닙니다.

Git에서 제외되는 환경 파일을 준비합니다.

```bash
cp .env.local-ai.example .env.local-ai
```

`CARD_HAS_MODEL_HOST`와 `INFANT_HEAD_TOP_CHECKPOINT_HOST`를 실제 호스트 파일로
설정하고, `H_ALIGN_AI_API_KEY`를 웹 Worker의 `AI_API_KEY`와 같은 값으로
설정합니다. 모델 파일과 API 키는 커밋하지 않습니다.

```bash
docker compose --env-file .env.local-ai -f compose.local-ai.yaml up -d
curl --fail http://127.0.0.1:8000/docs
docker logs macos-local-inference-ai-1
```

로컬 웹 Worker는 컨테이너에서 Mac 호스트로 접근하기 위해
`AI_BASE_URL=http://host.docker.internal:8000`을 사용합니다. 현재 CPU 검증
설정은 AI timeout 180초와 Queue visibility timeout 210초를 사용합니다.

2026-08-27 실제 모델 통합 테스트에서 다음을 확인했습니다.

- 카드 미검출: `200 + RETRY`, `CARD_NOT_FOUND`
- 유효한 입력: `200 + SUCCESS`, DB와 웹 결과 저장
- warm CPU 추론: 약 37초
- 내부 모델 경로와 디버그 정보가 일반 응답에 포함되지 않음

위 측정값과 시간은 통합 경로 검증 기록이며 정확도 또는 운영 성능을
보증하지 않습니다. 최초 실행은 ONNX 모델 다운로드와 모델 초기화 때문에
더 오래 걸리고 CPU 사용률이 높을 수 있습니다.

볼륨과 모델 파일을 삭제하지 않고 종료합니다.

```bash
docker compose --env-file .env.local-ai -f compose.local-ai.yaml stop
```

`down -v`는 사용하지 않습니다.

### Infant dataset merge and RTMPose fine-tuning

The current infant keypoint data can be merged from the extracted directories
under `dataset/infant_dataset_skel/splits`. The merger scans only immediate
directories whose names end with a part/range pattern such as
`part01_0001-0168`; ZIP files are ignored.

```bash
python scripts/merge_infant_keypoint_datasets.py
```

This creates `dataset/infant_dataset_skel/merged` with copied images and
`person_keypoints_train.json` / `person_keypoints_val.json`. When another
compatible directory is added to `splits`, run the same command again to
regenerate the complete merged dataset.

The height API preserves existing WholeBody joints and trains a separate
single-keypoint model only for `head_top`. Before a long run, generate the
derived one-keypoint annotations and validate the configuration:

```bash
python scripts/train_rtmpose_head_top.py \
  --dry-run \
  --device mps \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 0
```

Start the head_top-only fine-tuning run:

```bash
python scripts/train_rtmpose_head_top.py \
  --device mps \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 0 \
  --work-dir work_dirs/rtmpose_infant_head_top_only
```

Use `--checkpoint /path/to/checkpoint.pth` to continue from an existing
compatible head_top-only checkpoint. The old `train_rtmpose_infant.py` remains
available for experiments that intentionally re-train the complete 14-keypoint
schema. See
`docs/INFANT_RTMPOSE_TRAINING.md` for all options and validation details.
------------------------------------------------------------------------------------------------------------------------
