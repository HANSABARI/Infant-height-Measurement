# Windows GPU AI Server E2E Verification

이 문서는 Windows/NVIDIA PC에서 실제 AI 서버를 실행하고,
Cloudflare Quick Tunnel을 통해 `jaram-height-web` 백엔드/Worker가 호출할
HTTPS 경로를 검증한 결과를 팀에 공유하기 위한 기록이다.

## 요약

- 검증 일시: `2026-09-04 22:20 KST`
- AI 저장소 브랜치: `jaram-height-web`
- AI commit SHA: `360884631fb5cf0cf7b20f51e04d3c5c47168584`
- 운영 PC: Windows + NVIDIA GPU
- GPU: `NVIDIA GeForce RTX 4060 Ti`
- Python/Conda env: `jaram-height-web`
- 서버 포트: `127.0.0.1:8000`
- 임시 공개 Base URL: `https://soul-faced-crafts-extremely.trycloudflare.com`
- 검증 결과: Cloudflare HTTPS 경로로 `HTTP 200 + SUCCESS` 확인

Quick Tunnel URL은 임시 주소다. 터널 프로세스를 종료하거나 다시 만들면
주소가 바뀔 수 있으므로 운영 고정값으로 커밋하거나 장기 secret 설정에
박아두지 않는다.

## 연결 구조

```text
jaram-height-web Worker
  -> HTTPS Quick Tunnel URL
  -> Cloudflare Tunnel
  -> Windows GPU PC localhost:8000
  -> FastAPI AI server
  -> HaS card segmentation + RTMPose/WholeBody + head_top checkpoint
```

웹/Supabase 쪽에서 AI를 호출할 때 필요한 값은 두 개다.

- `AI_SERVER_URL`: Cloudflare가 발급한 공개 HTTPS base URL
- `AI_API_KEY`: AI 서버의 `H_ALIGN_AI_API_KEY`와 같은 Bearer secret

실제 secret 값은 Git, 공개 이슈, 공개 채팅에 기록하지 않는다.

## Cloudflare 경로와 API 인증 경계

Cloudflare Quick Tunnel은 FastAPI 서버 전체를 하나의 공개 HTTPS base URL로
연결한다.

```text
https://<quick-tunnel>.trycloudflare.com
  -> http://127.0.0.1:8000
```

따라서 endpoint마다 Cloudflare를 따로 붙이는 것이 아니다. FastAPI의 path를
그대로 뒤에 붙여 사용한다.

```text
http://127.0.0.1:8000/docs
  -> https://<quick-tunnel>.trycloudflare.com/docs

http://127.0.0.1:8000/api/v1/measure
  -> https://<quick-tunnel>.trycloudflare.com/api/v1/measure
```

Cloudflare tunnel token과 FastAPI API key는 역할이 다르다.

- Cloudflare tunnel token: 이 Windows PC의 `cloudflared`가 Cloudflare tunnel에
  연결할 수 있음을 인증한다.
- FastAPI Bearer API key: 외부 호출자가 `/api/v1/measure` 같은 AI API를 호출할
  수 있는지 서버 코드에서 검사한다.

즉 인증은 PC의 IP에 자동으로 붙는 것이 아니라, FastAPI의 각 endpoint 코드에서
직접 검사해야 한다. 현재 운영 endpoint인 `/api/v1/measure`는
`Authorization: Bearer <AI_API_KEY>`를 검사한다.

반면 `/api/v1/measure/debug`는 개발용 시각화 endpoint다. 기존 구조상 운영
계약 endpoint가 아니며, 이미지를 서버 디스크에 저장하고 내부 분석값을 더 많이
반환한다. Cloudflare Quick Tunnel을 통해 노출되면 인터넷에서 접근 가능한
업로드 endpoint가 되므로, 기본값에서는 다음처럼 처리한다.

- `H_ALIGN_ENABLE_DEBUG_API=1`일 때만 열림
- Swagger/OpenAPI 문서에는 표시하지 않음
- 일반 측정과 웹 연동 검증에는 사용하지 않음

Cloudflare까지 포함해 debug 저장을 검증해야 하는 경우에는 일시적으로만
`H_ALIGN_ENABLE_DEBUG_API=1`을 켜고, 검증 후 즉시 끈다. 장기 운영에서 debug
endpoint를 외부로 열어야 한다면 `/api/v1/measure`와 같은 Bearer 인증을 debug
endpoint에도 추가한 뒤 사용해야 한다.

## 모델과 런타임

- 카드 모델: `app/models/has_image_0209_fp32/sensitive_seg_best.pt`
- 정수리 checkpoint: `app/models/rtmpose_infant_head_top/best_coco_AP_epoch_85.pth`
- 모델 파일은 `.gitignore` 대상이며 Git에 커밋하지 않는다.
- `YOLO_CONFIG_DIR`가 없으면 Ultralytics 설정은 `app/.runtime/ultralytics`를 사용한다.
- 일반 운영에서는 `/api/v1/measure/debug`와 `debug_images` 저장을 켜지 않는다.

## 실행 절차

CMD에서 Quick Tunnel을 먼저 연다.

```cmd
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000
```

출력된 `https://*.trycloudflare.com` 주소를 검증용 `AI_SERVER_URL`로 사용한다.
이 터미널은 검증 중 계속 켜둔다.

PyCharm Git Bash에서 AI 서버를 실행한다.

```bash
cd /c/Users/ryuyoonmin/python_project/Infant-height-Measurement
git switch jaram-height-web
export H_ALIGN_AI_API_KEY="<shared-ai-api-key>"
bash scripts/run_ai_server_git_bash.sh
```

기대 로그:

```text
Uvicorn running on http://127.0.0.1:8000
Loading RTMPose-WholeBody model via rtmlib on CUDA
--- ✅ AI 모델 로딩 완료 ---
```

ONNX Runtime의 일부 CPU node 배정 경고는 CUDA 로딩 실패가 아니라 성능 관련
경고다. 실제 세션 provider는 `CUDAExecutionProvider`로 확인했다.

## 검증 명령과 결과

### 1. 로컬 헬스 체크

```bash
curl http://127.0.0.1:8000/
```

결과:

```json
{"message":"Hello Grow-Up Project!"}
```

### 2. Cloudflare HTTPS 헬스 체크

```bash
curl https://soul-faced-crafts-extremely.trycloudflare.com/
```

결과:

```json
{"message":"Hello Grow-Up Project!"}
```

### 3. SUCCESS 계약

```bash
curl -i -X POST https://soul-faced-crafts-extremely.trycloudflare.com/api/v1/measure \
  -H "Authorization: Bearer <shared-ai-api-key>" \
  -H "X-Measurement-Id: synthetic-success-test" \
  -F "file=@debug_images/synthetic_infant_card_validation.png;type=image/png"
```

결과:

```json
{
  "measurementId": "synthetic-success-test",
  "status": "SUCCESS",
  "result": {
    "estimatedHeightCm": 63.1,
    "heightRangeCm": null,
    "confidence": null,
    "quality": null,
    "modelVersion": "h-align-has-rtmpose-head-top-v1",
    "warnings": [],
    "measuredAt": "2026-09-04T13:20:07Z"
  },
  "error": null
}
```

확인한 계약:

- HTTP status는 `200 OK`
- 요청의 `X-Measurement-Id`가 `measurementId`로 그대로 반환됨
- `status`는 `SUCCESS`
- 검증 전 필드인 `heightRangeCm`, `confidence`, `quality`는 `null`
- 응답에 원본 이미지, 로컬 파일 경로, checkpoint 경로, stack trace가 없음

### 4. 인증 실패

```bash
curl -i -X POST https://soul-faced-crafts-extremely.trycloudflare.com/api/v1/measure \
  -H "X-Measurement-Id: auth-fail-test" \
  -F "file=not-an-image;filename=invalid.txt;type=text/plain"
```

기대 결과:

```text
HTTP/1.1 401 Unauthorized
```

응답에는 secret 값이 포함되지 않아야 한다.

### 5. RETRY 계약

```bash
curl -i -X POST https://soul-faced-crafts-extremely.trycloudflare.com/api/v1/measure \
  -H "Authorization: Bearer <shared-ai-api-key>" \
  -H "X-Measurement-Id: retry-test" \
  -F "file=not-an-image;filename=invalid.txt;type=text/plain"
```

기대 결과:

```json
{
  "measurementId": "retry-test",
  "status": "RETRY",
  "result": null,
  "error": {
    "code": "INVALID_IMAGE"
  }
}
```

카드가 없는 유효 이미지로도 `HTTP 200 + RETRY`와 `CARD_NOT_FOUND`를 확인했다.

### 6. FAILED 계약

head_top checkpoint를 없는 경로로 주입해 deterministic internal failure를
시뮬레이션했다.

결과:

```json
{
  "measurementId": "missing-checkpoint-local",
  "status": "FAILED",
  "result": null,
  "error": {
    "code": "INFERENCE_FAILED",
    "message": "AI 추론 모델을 준비하지 못했습니다."
  }
}
```

확인한 계약:

- HTTP status는 `200 OK`
- `status`는 `FAILED`
- 응답에 checkpoint 절대 경로가 포함되지 않음

## Debug 이미지

합성 검증 이미지와 실제 모델 추론 결과 overlay를 로컬에 저장했다.

- 입력 이미지: `debug_images/synthetic_infant_card_validation.png`
- debug overlay: `debug_images/synthetic_success_debug_visualization.jpg`

debug overlay 생성 결과:

```text
card_detected=True
card_conf=0.6838340759277344
px_per_cm=20.002083466431802
pose_detected=True
pose_conf=0.8987478613853455
height_cm=63.1
```

`debug_images/`는 Git ignored이므로, 이 이미지는 로컬 검증 증거로만 사용한다.
팀에 공유할 때는 필요하면 이미지 파일만 별도 전달하고, 사용자 실사진은 남기지
않는다.

## 이번에 발견하고 수정한 이슈

- PyCharm Git Bash에서는 Conda env의 `uvicorn`이 PATH에 안 잡힐 수 있어
  `D:/Anaconda3/envs/jaram-height-web/python.exe -m uvicorn ...` 방식으로 실행하도록
  스크립트를 추가했다.
- Windows에서 Ultralytics가 권한 없는 Roaming 설정 파일을 보다가 서버 시작이
  실패할 수 있어, `app/.runtime/ultralytics`를 기본 설정 경로로 잡았다.
- 한글/공백이 포함된 임시 API key 사용 시 `hmac.compare_digest()`가 500을 낼 수
  있어 UTF-8 bytes 기준 constant-time 비교로 수정했다.
- 기존 tracked 모델 가중치와 학습 로그는 파일은 유지하고 Git 추적에서 제거했다.

실운영 secret은 한글 문장보다 긴 랜덤 ASCII 문자열 사용을 권장한다.

## 자동 테스트

다음 명령으로 전체 테스트를 확인했다.

```bash
D:/Anaconda3/envs/jaram-height-web/python.exe -X utf8 -m unittest discover -s tests -v
```

결과:

```text
Ran 52 tests in 4.161s
OK
```

추가 확인:

```bash
D:/Anaconda3/envs/jaram-height-web/python.exe -X utf8 -m pip check
```

결과:

```text
No broken requirements found.
```

## 웹 팀에 전달할 내용

Quick Tunnel 검증 시:

```text
AI_SERVER_URL = https://soul-faced-crafts-extremely.trycloudflare.com
AI_API_KEY = 승인된 비밀 전달 수단으로 별도 공유
```

고정 운영 전환 시:

- Cloudflare 계정에 DNS zone을 추가한다.
- Named Tunnel route를 `ai.<domain>` 또는 `jaram-height-ai.<domain>`으로 잡는다.
- Worker/Supabase secret의 `AI_SERVER_URL`을 고정 HTTPS URL로 교체한다.
- `AI_API_KEY`는 AI 서버의 `H_ALIGN_AI_API_KEY`와 동일하게 맞춘다.

## 아직 남은 확인

- 실제 촬영 이미지로 `SUCCESS` 재검증
- 웹 저장소의 비동기 측정 Worker와 hosted Supabase queue 재시도 확인
- Windows AI 서버 재시작 후 대기 중인 새 측정 요청 처리 확인
- 고정 도메인 기반 Named Tunnel 구성
- GitHub issue `HANSABARI/jaram-height-web#3`에 검증 결과 연결

## GitHub 이슈 댓글 템플릿

```markdown
Windows GPU AI 서버 검증 결과

- AI commit SHA: 360884631fb5cf0cf7b20f51e04d3c5c47168584
- 공개 Base URL: https://soul-faced-crafts-extremely.trycloudflare.com
- 검증 일시: 2026-09-04 22:20 KST
- 담당자: <GitHub 사용자명>
- 환경: Windows, NVIDIA GeForce RTX 4060 Ti, Conda env jaram-height-web

검증:
- GET /: HTTP 200
- POST /api/v1/measure: HTTP 200 + SUCCESS
- X-Measurement-Id echo 확인
- heightRangeCm/confidence/quality null 확인
- 응답에 원본 이미지, 로컬 파일 경로, checkpoint 경로, stack trace 없음
- 인증 실패: HTTP 401
- RETRY: HTTP 200 + RETRY
- FAILED: HTTP 200 + FAILED

주의:
- 현재 Base URL은 Quick Tunnel 임시 URL이며 운영 고정 URL이 아님
- AI_API_KEY 및 tunnel token은 공개 이슈에 기록하지 않음
- 실제 촬영 이미지와 웹/Supabase queue E2E는 추가 검증 필요
```
