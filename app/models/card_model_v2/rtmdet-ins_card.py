## TO RUN MODEL
# conda activate rtmdet-ins
#
# python app/models/card_model_v2/rtmdet-ins_card.py \
#   --data-root "D:/PythonProject/nana-AI/app/dataset/card_instance_merged" \
#   --amp



import argparse
import os
import pathlib
import subprocess


_base_ = "mmdet::rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).replace("\\", "/")


classes = ("card",)
metainfo = dict(classes=classes)

data_root = _env("CARD_INSTANCE_DATA_ROOT", "app/dataset/card_instance")
train_ann = _env("CARD_INSTANCE_TRAIN_ANN", "annotations/instances_train.json")
val_ann = _env("CARD_INSTANCE_VAL_ANN", "annotations/instances_val.json")
img_prefix = _env("CARD_INSTANCE_IMG_PREFIX", "images/default/")

max_epochs = int(os.getenv("CARD_INSTANCE_EPOCHS", "100"))
stage2_num_epochs = int(os.getenv("CARD_INSTANCE_STAGE2_EPOCHS", "10"))
base_lr = float(os.getenv("CARD_INSTANCE_LR", "0.001"))
interval = int(os.getenv("CARD_INSTANCE_VAL_INTERVAL", "5"))
batch_size = int(os.getenv("CARD_INSTANCE_BATCH_SIZE", "8"))
num_workers = int(os.getenv("CARD_INSTANCE_NUM_WORKERS", "2"))

model = dict(
    bbox_head=dict(num_classes=1),
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    pin_memory=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann,
        data_prefix=dict(img=img_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann,
        data_prefix=dict(img=img_prefix),
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=str(pathlib.Path(data_root) / val_ann).replace("\\", "/"),
    metric=["bbox", "segm"],
)
test_evaluator = val_evaluator

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)],
)

optim_wrapper = dict(
    optimizer=dict(lr=base_lr),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

train_pipeline_stage2 = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type="RandomResize",
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(
        type="RandomCrop",
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type="PackDetInputs"),
]

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3,
        save_best="coco/segm_mAP",
        rule="greater",
    ),
)

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
    ),
]

env_cfg = dict(
    mp_cfg=dict(mp_start_method="spawn", opencv_num_threads=0),
)

load_from = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet-ins_tiny_8xb32-300e_coco/"
    "rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"
)
work_dir = "./app/models/card_model_rtmdet-ins"


def _set_training_env(options: argparse.Namespace) -> None:
    os.environ["CARD_INSTANCE_DATA_ROOT"] = str(pathlib.Path(options.data_root)).replace("\\", "/")
    os.environ["CARD_INSTANCE_TRAIN_ANN"] = options.train_ann.replace("\\", "/")
    os.environ["CARD_INSTANCE_VAL_ANN"] = options.val_ann.replace("\\", "/")
    os.environ["CARD_INSTANCE_IMG_PREFIX"] = options.img_prefix.replace("\\", "/")
    os.environ["CARD_INSTANCE_EPOCHS"] = str(options.epochs)
    os.environ["CARD_INSTANCE_STAGE2_EPOCHS"] = str(options.stage2_epochs)
    os.environ["CARD_INSTANCE_LR"] = str(options.lr)
    os.environ["CARD_INSTANCE_VAL_INTERVAL"] = str(options.val_interval)
    os.environ["CARD_INSTANCE_BATCH_SIZE"] = str(options.batch_size)
    os.environ["CARD_INSTANCE_NUM_WORKERS"] = str(options.num_workers)


def _validate_dataset_paths(options: argparse.Namespace) -> None:
    data_root_path = pathlib.Path(options.data_root)
    train_ann_path = data_root_path / options.train_ann
    val_ann_path = data_root_path / options.val_ann
    img_dir = data_root_path / options.img_prefix

    missing = [
        path
        for path in (data_root_path, train_ann_path, val_ann_path, img_dir)
        if not path.exists()
    ]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Dataset path check failed:\n{joined}")


def _dry_run(resolved_config_path: pathlib.Path) -> None:
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.registry import DATASETS

    cfg = Config.fromfile(str(resolved_config_path))
    init_default_scope("mmdet")
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    sample = dataset[0]
    gt_instances = sample["data_samples"].gt_instances

    print(f"config ok: {resolved_config_path}")
    print(f"train samples: {len(dataset)}")
    print(f"input shape: {tuple(sample['inputs'].shape)}")
    print(f"bbox shape: {tuple(gt_instances.bboxes.shape)}")
    print(f"mask type: {type(gt_instances.masks).__name__}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RTMDet-Ins for card instance segmentation.")
    parser.add_argument("--data-root", required=True, help="COCO dataset root.")
    parser.add_argument("--train-ann", default="annotations/instances_train.json")
    parser.add_argument("--val-ann", default="annotations/instances_val.json")
    parser.add_argument("--img-prefix", default="images/default/")
    parser.add_argument("--work-dir", default="./app/models/card_model_rtmdet-ins")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dataset loading only.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_options = _parse_args()
    _validate_dataset_paths(cli_options)
    _set_training_env(cli_options)

    resolved_config_path = pathlib.Path(__file__).resolve()
    if cli_options.dry_run:
        _dry_run(resolved_config_path)
        raise SystemExit(0)

    command = [
        "mim",
        "train",
        "mmdet",
        str(resolved_config_path),
        "--work-dir",
        cli_options.work_dir,
        "--gpus",
        str(cli_options.gpus),
        "--launcher",
        "none",
        "--yes",
    ]
    if cli_options.amp:
        command.append("--amp")

    raise SystemExit(subprocess.run(command, check=False).returncode)
