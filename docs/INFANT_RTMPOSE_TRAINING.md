# Infant RTMPose Dataset and Fine-tuning

## Scripts

- Merge extracted datasets: `scripts/merge_infant_keypoint_datasets.py`
- Train `head_top` only: `scripts/train_rtmpose_head_top.py`
- Optional full 14-keypoint experiment: `scripts/train_rtmpose_infant.py`
- Output dataset: `dataset/infant_dataset_skel/merged`

## Dataset merge

The merger checks only immediate child directories of:

```text
dataset/infant_dataset_skel/splits
```

A directory is accepted when its name ends with a pattern such as
`part01_0001-0168`. The prefix can change, so names such as
`명준_infant_dataset_skel_part02_0169-0336` and
`윤민_infant_dataset_skel_part01_0001-0168` are both accepted. ZIP files,
`.DS_Store`, and unrelated directories are ignored.

Run from the project root:

```bash
python scripts/merge_infant_keypoint_datasets.py
```

Optional arguments:

```bash
python scripts/merge_infant_keypoint_datasets.py \
  --splits-dir dataset/infant_dataset_skel/splits \
  --output-root dataset/infant_dataset_skel/merged \
  --val-ratio 0.2 \
  --seed 42
```

The script creates:

```text
dataset/infant_dataset_skel/merged/
├── annotations/
│   ├── person_keypoints.json
│   ├── person_keypoints_train.json
│   └── person_keypoints_val.json
├── images/
└── manifest.json
```

Images are copied with a source-directory prefix, and COCO image/annotation
IDs are regenerated. The 14 keypoints are normalized to this order:

```text
nose, left_shoulder, right_shoulder, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle, left_eye, right_eye,
left_heel, right_heel, head_top
```

`manifest.json` records source counts and invisible keypoint counts. An
invisible point remains `(0, 0, 0)`; the merger does not silently remove the
image. Running the same merge command again regenerates the complete output,
including any newly added compatible source directory.

## Head Top-Only Fine-Tuning

The active measurement pipeline does not re-train the existing WholeBody
keypoints. `scripts/train_rtmpose_head_top.py` derives COCO annotation views
containing only visible `head_top` labels, then trains a one-output RTMPose-M
model. At inference, its one predicted point is combined with the original
WholeBody `nose`, shoulders, hips, knees, ankles, eyes, and heels.

First check the generated runtime config:

```bash
python scripts/train_rtmpose_head_top.py \
  --dry-run \
  --device mps \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 0 \
  --work-dir /tmp/rtmpose-head-top-dry-run
```

Then start training:

```bash
python scripts/train_rtmpose_head_top.py \
  --device mps \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 0 \
  --work-dir work_dirs/rtmpose_infant_head_top_only
```

Useful options:

- `--dataset-root`: merged dataset root; default is `dataset/infant_dataset_skel/merged`.
- `--checkpoint`: optional `.pth` checkpoint to fine-tune from.
- `--base-config`: optional custom MMPose config; otherwise the installed RTMPose-M COCO config is used.
- `--epochs`: training epochs; default `100`.
- `--batch-size`: train/validation batch size; default `8`.
- `--num-workers`: data loader workers; default `4`.
- `--device`: `cpu`, `cuda`, or `mps` depending on the PyTorch environment.
- `--val-interval`: validation interval in epochs; default `5`.
- `--seed`: deterministic split/config seed; default `42`.
- `--learning-rate`: final optimizer LR. By default, the RTMPose reference LR
  is linearly scaled from its batch-1024 reference to the selected batch size.
- `--warmup-epochs`: maximum warmup duration; default `5`.

The training environment must provide `mmpose`, `mmengine`, `torch`, and
`albumentations==1.4.24`.
The derived annotations are written beside the merged annotations as
`person_keypoints_head_top_train.json` and
`person_keypoints_head_top_val.json`. Training output and the resolved
`rtmpose_head_top_runtime.py` are written under `--work-dir`. The server does
not depend on that generated config: it uses the tracked inference config at
`app/configs/rtmpose_head_top.py` by default.

## Inference checkpoint

The `/api/v1/measure` endpoint loads the best local head_top-only checkpoint
by default:

```text
work_dirs/rtmpose_infant_head_top_only/best_coco_AP_epoch_*.pth
```

The best epoch is selected automatically, so its filename does not need to end
in `100`.

WholeBody supplies the existing 13 points. The head_top-only model uses the
WholeBody person box and contributes only `head_top`; height is calculated as
`head_top -> shoulder center -> hip center -> the longer hip-knee-ankle-heel
chain`, converted with the card's `px_per_cm`. No fixed head or foot-length
correction is added.

For a different local checkpoint or runtime config, set these before starting
the server. The checkpoint remains a local deployment artifact and is not
committed to Git.

```bash
export INFANT_HEAD_TOP_CHECKPOINT=/absolute/path/to/checkpoint.pth
export INFANT_HEAD_TOP_CONFIG=/absolute/path/to/rtmpose_head_top.py
export INFANT_HEAD_TOP_DEVICE=mps
```
