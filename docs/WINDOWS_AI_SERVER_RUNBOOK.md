# Windows AI Server Runbook

This runbook is for running the AI server on the Windows/NVIDIA PC and exposing it to `jaram-height-web` through Cloudflare Tunnel.

## Current Local Layout

- Project: `C:\Users\ryuyoonmin\python_project\Infant-height-Measurement`
- Conda env: `D:\Anaconda3\envs\jaram-height-web`
- Card weights: `app\models\has_image_0209_fp32\sensitive_seg_best.pt`
- Head-top checkpoint: `app\models\rtmpose_infant_head_top\best_coco_AP_epoch_85.pth`

Model files and real secrets must stay out of Git.

## Start The AI Server

Git Bash from PyCharm may not put the Conda env `Scripts` directory on `PATH`, so prefer the repo script or `python -m uvicorn`.

Git Bash:

```bash
cd /c/Users/ryuyoonmin/python_project/Infant-height-Measurement
export H_ALIGN_AI_API_KEY="<shared-ai-api-key>"
bash scripts/run_ai_server_git_bash.sh
```

Direct Git Bash fallback:

```bash
cd /c/Users/ryuyoonmin/python_project/Infant-height-Measurement
export H_ALIGN_AI_API_KEY="<shared-ai-api-key>"
D:/Anaconda3/envs/jaram-height-web/python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

PowerShell:

```powershell
cd C:\Users\ryuyoonmin\python_project\Infant-height-Measurement
$env:H_ALIGN_AI_API_KEY = "<shared-ai-api-key>"
.\scripts\run_ai_server_windows.ps1 -Reload
```

## Cloudflare Tunnel

For a temporary Quick Tunnel:

```cmd
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:8000
```

Keep that terminal open. The printed `https://*.trycloudflare.com` URL forwards to `http://127.0.0.1:8000` on this PC.

For a stable production URL, the Cloudflare account needs a DNS zone. Add a published application route:

- Public hostname: `ai.<domain>` or `jaram-height-ai.<domain>`
- Path: empty
- Service URL: `http://localhost:8000`

The web side should use the public HTTPS base URL as `AI_SERVER_URL` and the same shared secret as `AI_API_KEY`.

The tunnel URL maps to the whole FastAPI app, not to a single endpoint. Use one
Cloudflare base URL and append FastAPI paths:

```text
https://<public-ai-host>/docs
https://<public-ai-host>/api/v1/measure
```

Cloudflare tunnel authentication only allows this Windows PC to connect as the
tunnel origin. It does not authenticate callers of individual FastAPI endpoints.
Caller authentication must be enforced in FastAPI endpoint code with
`Authorization: Bearer <AI_API_KEY>`.

## Operational Defaults

- `H_ALIGN_AI_API_KEY` is required for `/api/v1/measure`.
- `/api/v1/measure/debug` is disabled unless `H_ALIGN_ENABLE_DEBUG_API=1`.
- `/api/v1/measure/debug` is a development endpoint for saving local visual
  overlays. It is hidden from Swagger/OpenAPI and should not be enabled for
  ordinary measurements.
- If the debug endpoint ever needs to be exposed through Cloudflare for more
  than a short manual check, add the same Bearer authentication used by
  `/api/v1/measure` before exposing it.
- If `YOLO_CONFIG_DIR` is not set, the server uses
  `app\.runtime\ultralytics` for Ultralytics settings. This local runtime
  cache is Git ignored.
- Do not share Cloudflare tunnel tokens, API keys, local paths, or checkpoint paths in public issues or chat.

## Health Checks

Local:

```bash
curl http://127.0.0.1:8000/
```

Tunnel:

```bash
curl https://<public-ai-host>/
```

Expected root response:

```json
{"message":"Hello Grow-Up Project!"}
```

## Restart

1. Start or verify Cloudflare:

```cmd
sc query Cloudflared
sc start Cloudflared
```

2. Start the AI server with the commands above.
3. Verify `/` locally and through the tunnel.
4. Then run authenticated `/api/v1/measure` contract tests from the web integration side.

## 2026-09-04 Windows GPU Verification

- Branch: `jaram-height-web`
- AI commit SHA: `360884631fb5cf0cf7b20f51e04d3c5c47168584`
- Verification time: `2026-09-04 22:20 KST`
- Owner: `<GitHub username required>`
- Temporary public base URL: `https://soul-faced-crafts-extremely.trycloudflare.com`
- Python: `D:\Anaconda3\envs\jaram-height-web\python.exe`
- GPU: `NVIDIA GeForce RTX 4060 Ti`
- NVIDIA driver: `610.88`
- CUDA runtime seen by PyTorch: `12.1`
- NVIDIA-SMI CUDA UMD: `13.3`
- PyTorch: `2.1.0+cu121`
- ONNX Runtime: `1.18.0`
- ONNX Runtime providers: `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`
- Cloudflare Windows service: `Cloudflared`, running

Verified:

- `pip check` reported no broken requirements.
- The card weights and head-top checkpoint exist in `app\models`.
- The API imports real HaS card detection and RTMPose/WholeBody components.
- RTMPose/WholeBody ONNX sessions use `CUDAExecutionProvider`.
- Local root health check returned `HTTP 200` and `{"message":"Hello Grow-Up Project!"}`.
- The temporary Cloudflare URL returned the same root health response over HTTPS.
- A synthetic privacy-safe infant/card image returned `HTTP 200` + `SUCCESS`
  through both local and Cloudflare HTTPS paths.
- A valid no-card image returned `HTTP 200` + `RETRY` with `CARD_NOT_FOUND`
  through both local and Cloudflare HTTPS paths.
- A request without the Bearer API key returned `HTTP 401` without secrets.
- A missing head-top checkpoint simulation returned `HTTP 200` + `FAILED`
  with `INFERENCE_FAILED` and no checkpoint path in the body.
- Unit tests passed: `python -X utf8 -m unittest discover -s tests -v` ran 52 tests with `OK`.

Still requires an external artifact or account state:

- A real captured infant/card test image should still be used before production,
  even though the synthetic image proves the live `SUCCESS` contract.
- A stable named Cloudflare hostname requires a Cloudflare DNS zone in the account. Without a zone, use the temporary `trycloudflare.com` Quick Tunnel URL only for testing.
- GitHub issue `HANSABARI/jaram-height-web#3` must be checked from an authenticated GitHub session if the repository or issue is private.
- Web/Supabase queue retry behavior must be verified from the web repository with the approved `AI_SERVER_URL` and `AI_API_KEY` secrets.
