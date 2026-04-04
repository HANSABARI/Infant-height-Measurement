# 📏 h-align-server (nana-AI)

------------------------------------------------------------------------------------------------------------------------
An AI-powered FastAPI backend server that calculates a person's actual height from an image. 
The pipeline relies on detecting a standard-sized reference object (a credit card) and extracting human body keypoints.

## 🌟 Core AI Pipeline
1. **Card Detection (RTMDet):** Detects a reference card (8.56cm x 5.4cm) to calculate the `pixel-per-cm` ratio.
2. **Pose Estimation (RTMPose):** Extracts human body keypoints (skeleton) using ONNX runtime (`rtmlib`).
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
│   │   ├── card_detector_rtm.py  # RTMDet Inference (w/ PyTorch security patch)
│   │   ├── pose_estimator.py     # RTMPose Inference
│   │   ├── height_calculator.py  # Physical height calculation logic
│   │   └── visualizer.py         # Drawing bounding boxes and skeletons
│   └── models/                   # Pre-trained weights and config files
│       └── card_model_v2/        
│           ├── rtmdet_nano_card.py # Final RTMDet config
│           └── epoch_50.pth      # Fine-tuned RTMDet weights (50 epochs)
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

🚨 Known Issues & Workarounds
1. PyTorch 2.6+ Security Policy (torch.load UnpicklingError)
Starting from PyTorch 2.6, the default behavior of torch.load has been restricted (weights_only=True by default) for security reasons.
Loading OpenMMLab weights (.pth) will trigger an UnpicklingError due to embedded numpy objects and history buffers.

Solution Applied: A monkey-patching workaround (weights_only=False) is already implemented inside app/services/card_detector_rtm.py.
No manual action is required to run the server.


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

Main Endpoint: POST /api/v1/measure

Debug Endpoint (Saves image locally): POST /api/v1/measure/debug
------------------------------------------------------------------------------------------------------------------------