#!/usr/bin/env python3
"""Fine-tune MMPose RTMPose on the merged infant keypoint dataset."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


KEYPOINT_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_eye",
    "right_eye",
    "left_heel",
    "right_heel",
    "head_top",
]

SKELETON_LINKS = [
    ("left_eye", "nose"),
    ("right_eye", "nose"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
]

DEFAULT_WARMUP_EPOCHS = 5

SWAPS = {
    "left_shoulder": "right_shoulder",
    "right_shoulder": "left_shoulder",
    "left_hip": "right_hip",
    "right_hip": "left_hip",
    "left_knee": "right_knee",
    "right_knee": "left_knee",
    "left_ankle": "right_ankle",
    "right_ankle": "left_ankle",
    "left_eye": "right_eye",
    "right_eye": "left_eye",
    "left_heel": "right_heel",
    "right_heel": "left_heel",
}


def build_pose_metainfo(
    keypoint_names: list[str] = KEYPOINT_NAMES,
    skeleton_links: list[tuple[str, str]] = SKELETON_LINKS,
    dataset_name: str = "infant14",
) -> dict[str, Any]:
    """Return MMPose metainfo for the custom infant keypoint schema."""
    keypoint_info = {}
    for index, name in enumerate(keypoint_names):
        keypoint_info[index] = {
            "name": name,
            "type": "lower" if any(token in name for token in ("hip", "knee", "ankle", "heel")) else "upper",
            "swap": SWAPS.get(name, name),
            "color": [51, 153, 255],
        }

    skeleton_info = {
        index: {"link": link, "color": [96, 96, 255]}
        for index, link in enumerate(skeleton_links)
    }
    return {
        "dataset_name": dataset_name,
        "keypoint_info": keypoint_info,
        "skeleton_info": skeleton_info,
        "joint_weights": [1.0] * len(keypoint_names),
        "sigmas": [0.05] * len(keypoint_names),
    }


def find_default_base_config() -> Path:
    """Find MMPose's official RTMPose-M COCO config in the active environment."""
    try:
        import mmpose
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MMPose is not installed. Install mmpose, mmengine, and torch in the "
            "training environment before running this script."
        ) from exc

    config = (
        Path(mmpose.__file__).resolve().parent
        / ".mim"
        / "configs"
        / "body_2d_keypoint"
        / "rtmpose"
        / "coco"
        / "rtmpose-m_8xb256-420e_coco-256x192.py"
    )
    if not config.is_file():
        raise FileNotFoundError(f"Could not find the default MMPose config: {config}")
    return config


def configure_mmengine_device(device: str) -> None:
    """Force MMEngine to use the device selected by the command-line option."""
    normalized = device.lower()
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: cpu, cuda, mps")

    # MMEngine 0.10 chooses a global device at import time. Its Runner ignores
    # cfg.env_cfg.device, so update that selector before creating the Runner.
    from mmengine.device import utils as device_utils

    device_utils.DEVICE = normalized


def enable_mps_rtmcc_accuracy_compatibility() -> None:
    """Patch MMPose 1.3 RTMCC accuracy logging for MPS float32 support."""
    import torch
    from mmpose.evaluation.functional import simcc_pck_accuracy
    from mmpose.models.heads.coord_cls_heads.rtmcc_head import RTMCCHead
    from mmpose.utils.tensor_utils import to_numpy

    if getattr(RTMCCHead.loss, "_infant_mps_compatibility", False):
        return

    def loss(self, feats, batch_data_samples, train_cfg={}):
        pred_x, pred_y = self.forward(feats)
        gt_x = torch.cat(
            [sample.gt_instance_labels.keypoint_x_labels for sample in batch_data_samples],
            dim=0,
        )
        gt_y = torch.cat(
            [sample.gt_instance_labels.keypoint_y_labels for sample in batch_data_samples],
            dim=0,
        )
        keypoint_weights = torch.cat(
            [sample.gt_instance_labels.keypoint_weights for sample in batch_data_samples],
            dim=0,
        )
        losses = {
            "loss_kpt": self.loss_module(
                (pred_x, pred_y), (gt_x, gt_y), keypoint_weights
            )
        }
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy((pred_x, pred_y)),
            target=to_numpy((gt_x, gt_y)),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )
        # `torch.tensor(float)` defaults to float64, unsupported by MPS.
        losses["acc_pose"] = gt_x.new_tensor(avg_acc, dtype=torch.float32)
        return losses

    loss._infant_mps_compatibility = True
    RTMCCHead.loss = loss


def _set_dataset_config(
    dataset: Any,
    dataset_root: Path,
    annotation_name: str,
    pose_metainfo: dict[str, Any],
) -> None:
    dataset.data_root = str(dataset_root.resolve()) + "/"
    dataset.ann_file = f"annotations/{annotation_name}"
    dataset.data_prefix = dict(img="images/")
    dataset.metainfo = copy.deepcopy(pose_metainfo)


def _train_iterations_per_epoch(
    dataset_root: Path,
    batch_size: int,
    train_annotation_name: str = "person_keypoints_train.json",
) -> int:
    annotation_path = dataset_root / "annotations" / train_annotation_name
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Training annotation file not found: {annotation_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid training annotation JSON: {annotation_path}") from exc

    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"Training annotation has no images: {annotation_path}")
    return math.ceil(len(images) / batch_size)


def _configure_optimization_for_batch(
    cfg: Any,
    dataset_root: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float | None,
    warmup_epochs: int,
    train_annotation_name: str,
) -> None:
    """Scale RTMPose's reference-batch schedule to this training run."""
    if learning_rate is not None and learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if warmup_epochs <= 0:
        raise ValueError("warmup_epochs must be greater than 0")

    reference_batch_size = int(cfg.get("auto_scale_lr", {}).get("base_batch_size", batch_size))
    if reference_batch_size <= 0:
        raise ValueError("auto_scale_lr.base_batch_size must be greater than 0")

    reference_lr = float(cfg.optim_wrapper.optimizer.lr)
    effective_lr = (
        float(learning_rate)
        if learning_rate is not None
        else reference_lr * batch_size / reference_batch_size
    )
    lr_scale = effective_lr / reference_lr
    cfg.optim_wrapper.optimizer.lr = effective_lr
    # The optimizer and scheduler are scaled here together. Leave MMEngine's
    # later auto-scaling disabled so it cannot scale the optimizer a second time.
    cfg.auto_scale_lr = dict(base_batch_size=reference_batch_size, enable=False)

    warmup_steps = None
    for scheduler in cfg.param_scheduler:
        scheduler_type = scheduler.get("type")
        if scheduler_type == "LinearLR":
            if warmup_steps is None:
                iterations_per_epoch = _train_iterations_per_epoch(
                    dataset_root,
                    batch_size,
                    train_annotation_name,
                )
                warmup_steps = min(
                    int(scheduler.end),
                    iterations_per_epoch * min(warmup_epochs, epochs),
                )
            scheduler.end = warmup_steps
        elif scheduler_type == "CosineAnnealingLR":
            scheduler.eta_min = float(scheduler.eta_min) * lr_scale


def build_runtime_config(
    base_config: Path,
    dataset_root: Path,
    work_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: str,
    checkpoint: Path | None,
    val_interval: int,
    seed: int,
    learning_rate: float | None = None,
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
    keypoint_names: list[str] = KEYPOINT_NAMES,
    skeleton_links: list[tuple[str, str]] = SKELETON_LINKS,
    dataset_name: str = "infant14",
    train_annotation_name: str = "person_keypoints_train.json",
    val_annotation_name: str = "person_keypoints_val.json",
):
    """Load the official config and override it for the merged infant dataset."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if num_workers < 0:
        raise ValueError("num_workers must be greater than or equal to 0")
    if val_interval <= 0:
        raise ValueError("val_interval must be greater than 0")

    try:
        from mmengine import Config
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MMEngine is not installed. Install mmengine in the training environment."
        ) from exc

    cfg = Config.fromfile(str(base_config))
    cfg.work_dir = str(work_dir.resolve())
    cfg.train_cfg.max_epochs = epochs
    cfg.train_cfg.val_interval = val_interval
    cfg.randomness = dict(cfg.get("randomness", {}), seed=seed)
    cfg.env_cfg.device = device
    cfg.device = device

    cfg.model.head.out_channels = len(keypoint_names)
    cfg.model.head.input_size = tuple(cfg.codec.input_size)
    cfg.model.head.in_featuremap_size = tuple(size // 32 for size in cfg.codec.input_size)
    cfg.model.head.simcc_split_ratio = cfg.codec.simcc_split_ratio
    cfg.model.head.decoder = copy.deepcopy(cfg.codec)

    dataset_root = dataset_root.resolve()
    pose_metainfo = build_pose_metainfo(
        keypoint_names=keypoint_names,
        skeleton_links=skeleton_links,
        dataset_name=dataset_name,
    )
    _set_dataset_config(
        cfg.train_dataloader.dataset,
        dataset_root,
        train_annotation_name,
        pose_metainfo,
    )
    _set_dataset_config(
        cfg.val_dataloader.dataset,
        dataset_root,
        val_annotation_name,
        pose_metainfo,
    )
    _set_dataset_config(
        cfg.test_dataloader.dataset,
        dataset_root,
        val_annotation_name,
        pose_metainfo,
    )

    cfg.train_dataloader.batch_size = batch_size
    cfg.val_dataloader.batch_size = batch_size
    cfg.test_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = num_workers
    cfg.val_dataloader.num_workers = num_workers
    cfg.test_dataloader.num_workers = num_workers
    persistent_workers = num_workers > 0
    cfg.train_dataloader.persistent_workers = persistent_workers
    cfg.val_dataloader.persistent_workers = persistent_workers
    cfg.test_dataloader.persistent_workers = persistent_workers

    val_annotation = str(dataset_root / "annotations" / val_annotation_name)
    cfg.val_evaluator.ann_file = val_annotation
    cfg.test_evaluator.ann_file = val_annotation

    if checkpoint is not None:
        cfg.load_from = str(checkpoint.resolve())
    elif "load_from" in cfg:
        cfg.load_from = None

    if "stage2_num_epochs" in cfg:
        cfg.stage2_num_epochs = min(int(cfg.stage2_num_epochs), max(1, epochs // 10))
        cfg.custom_hooks[1].switch_epoch = max(0, epochs - cfg.stage2_num_epochs)
    if epochs < 5:
        # The default warmup spans 1,000 iterations, which can outlast a
        # short smoke run and leave the cosine schedule with an empty range.
        cfg.param_scheduler = [
            dict(
                type="CosineAnnealingLR",
                T_max=epochs,
                eta_min=0.0002,
                by_epoch=True,
                begin=0,
                end=epochs,
            )
        ]
    elif len(cfg.param_scheduler) > 1:
        cfg.param_scheduler[1].begin = max(1, epochs // 2)
        cfg.param_scheduler[1].end = epochs
        cfg.param_scheduler[1].T_max = max(1, epochs // 2)

    _configure_optimization_for_batch(
        cfg=cfg,
        dataset_root=dataset_root,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        warmup_epochs=warmup_epochs,
        train_annotation_name=train_annotation_name,
    )

    return cfg


def validate_dataset_root(
    dataset_root: Path,
    train_annotation_name: str = "person_keypoints_train.json",
    val_annotation_name: str = "person_keypoints_val.json",
) -> None:
    required = [
        dataset_root / "images",
        dataset_root / "annotations" / train_annotation_name,
        dataset_root / "annotations" / val_annotation_name,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Merged dataset is incomplete. Missing: " + ", ".join(missing)
        )


def write_runtime_config(
    cfg: Any,
    work_dir: Path,
    filename: str = "rtmpose_infant_runtime.py",
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / filename
    config_path.write_text(cfg.pretty_text, encoding="utf-8")
    return config_path


def run_training(cfg: Any) -> None:
    try:
        configure_mmengine_device(str(cfg.device))
        from mmpose.utils import register_all_modules
        from mmengine.runner import Runner
        register_all_modules(init_default_scope=True)
        if str(cfg.device).lower() == "mps":
            enable_mps_rtmcc_accuracy_compatibility()
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "RTMPose training dependencies could not be imported. Install compatible "
            "versions of mmpose, mmengine, mmcv, and torch in the training environment; "
            "an mmcv/_ext symbol error usually means mmcv was built for a different "
            "PyTorch version."
        ) from exc

    runner = Runner.from_cfg(cfg)
    runner.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune RTMPose on the merged infant keypoint dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/infant_dataset_skel/merged"),
    )
    parser.add_argument("--work-dir", type=Path, default=Path("work_dirs/rtmpose_infant"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Final optimizer learning rate. Defaults to linear scaling from batch 1024.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help=f"Maximum warmup duration in epochs. Default: {DEFAULT_WARMUP_EPOCHS}.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    work_dir = args.work_dir.resolve()
    validate_dataset_root(dataset_root)
    base_config = args.base_config.resolve() if args.base_config else find_default_base_config()
    cfg = build_runtime_config(
        base_config=base_config,
        dataset_root=dataset_root,
        work_dir=work_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        checkpoint=args.checkpoint,
        val_interval=args.val_interval,
        seed=args.seed,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
    )
    config_path = write_runtime_config(cfg, work_dir)
    print(f"base_config: {base_config}")
    print(f"runtime_config: {config_path}")
    print(f"dataset_root: {dataset_root}")
    lr_label = args.learning_rate if args.learning_rate is not None else "auto"
    print(
        f"epochs: {args.epochs}, batch_size: {args.batch_size}, device: {args.device}, "
        f"learning_rate: {lr_label}, warmup_epochs: {args.warmup_epochs}"
    )
    if args.dry_run:
        print("dry_run: training was not started")
        return
    run_training(cfg)


if __name__ == "__main__":
    main()
