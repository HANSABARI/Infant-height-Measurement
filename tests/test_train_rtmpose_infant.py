import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_rtmpose_infant.py"
spec = importlib.util.spec_from_file_location("train_rtmpose_infant", SCRIPT_PATH)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)


class TrainRtmposeInfantTests(unittest.TestCase):
    def test_pose_metainfo_has_fourteen_keypoints_and_symmetric_pairs(self):
        metainfo = train_module.build_pose_metainfo()

        self.assertEqual(len(metainfo["keypoint_info"]), 14)
        self.assertEqual(metainfo["keypoint_info"][13]["name"], "head_top")
        self.assertEqual(metainfo["keypoint_info"][1]["swap"], "right_shoulder")
        self.assertEqual(metainfo["keypoint_info"][13]["swap"], "head_top")
        self.assertEqual(len(metainfo["joint_weights"]), 14)
        self.assertEqual(len(metainfo["sigmas"]), 14)

    def test_runtime_config_uses_merged_dataset_and_fourteen_keypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation_dir = root / "merged" / "annotations"
            annotation_dir.mkdir(parents=True)
            (annotation_dir / "person_keypoints_train.json").write_text(
                json.dumps({"images": [{"id": index} for index in range(268)]}),
                encoding="utf-8",
            )
            cfg = train_module.build_runtime_config(
                base_config=train_module.find_default_base_config(),
                dataset_root=root / "merged",
                work_dir=root / "work",
                epochs=12,
                batch_size=4,
                num_workers=0,
                device="cpu",
                checkpoint=root / "checkpoint.pth",
                val_interval=3,
                seed=7,
            )

        self.assertEqual(cfg.model.head.out_channels, 14)
        self.assertEqual(cfg.train_cfg.max_epochs, 12)
        self.assertEqual(cfg.train_cfg.val_interval, 3)
        self.assertEqual(cfg.train_dataloader.batch_size, 4)
        self.assertEqual(cfg.train_dataloader.num_workers, 0)
        self.assertFalse(cfg.train_dataloader.persistent_workers)
        self.assertEqual(
            cfg.train_dataloader.dataset.ann_file,
            "annotations/person_keypoints_train.json",
        )
        self.assertEqual(cfg.val_dataloader.dataset.data_prefix.img, "images/")
        self.assertEqual(cfg.val_dataloader.dataset.ann_file, "annotations/person_keypoints_val.json")
        self.assertEqual(cfg.env_cfg.device, "cpu")
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.randomness.seed, 7)
        self.assertTrue(str(cfg.load_from).endswith("checkpoint.pth"))

    def test_one_epoch_runtime_config_uses_a_valid_short_run_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = train_module.build_runtime_config(
                base_config=train_module.find_default_base_config(),
                dataset_root=root / "merged",
                work_dir=root / "work",
                epochs=1,
                batch_size=1,
                num_workers=0,
                device="cpu",
                checkpoint=None,
                val_interval=1,
                seed=42,
            )

        self.assertEqual(len(cfg.param_scheduler), 1)
        self.assertEqual(cfg.param_scheduler[0].type, "CosineAnnealingLR")
        self.assertEqual(cfg.param_scheduler[0].begin, 0)
        self.assertEqual(cfg.param_scheduler[0].end, 1)
        self.assertEqual(cfg.param_scheduler[0].T_max, 1)

    def test_runtime_config_scales_optimization_for_small_batch_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "merged"
            annotation_dir = dataset_root / "annotations"
            annotation_dir.mkdir(parents=True)
            (annotation_dir / "person_keypoints_train.json").write_text(
                json.dumps({"images": [{"id": index} for index in range(268)]}),
                encoding="utf-8",
            )
            cfg = train_module.build_runtime_config(
                base_config=train_module.find_default_base_config(),
                dataset_root=dataset_root,
                work_dir=root / "work",
                epochs=100,
                batch_size=8,
                num_workers=0,
                device="cpu",
                checkpoint=None,
                val_interval=5,
                seed=42,
            )

        linear_scheduler, cosine_scheduler = cfg.param_scheduler
        self.assertAlmostEqual(cfg.optim_wrapper.optimizer.lr, 0.00003125)
        self.assertAlmostEqual(cosine_scheduler.eta_min, 0.0000015625)
        self.assertEqual(linear_scheduler.end, 170)

    def test_runtime_config_uses_explicit_learning_rate_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "merged"
            annotation_dir = dataset_root / "annotations"
            annotation_dir.mkdir(parents=True)
            (annotation_dir / "person_keypoints_train.json").write_text(
                json.dumps({"images": [{"id": index} for index in range(8)]}),
                encoding="utf-8",
            )
            cfg = train_module.build_runtime_config(
                base_config=train_module.find_default_base_config(),
                dataset_root=dataset_root,
                work_dir=root / "work",
                epochs=100,
                batch_size=8,
                num_workers=0,
                device="cpu",
                checkpoint=None,
                val_interval=5,
                seed=42,
                learning_rate=0.0001,
            )

        _, cosine_scheduler = cfg.param_scheduler
        self.assertAlmostEqual(cfg.optim_wrapper.optimizer.lr, 0.0001)
        self.assertAlmostEqual(cosine_scheduler.eta_min, 0.000005)

    def test_configure_mmengine_device_honors_cpu(self):
        from mmengine.device import utils as device_utils

        original_device = device_utils.DEVICE
        try:
            train_module.configure_mmengine_device("cpu")
            self.assertEqual(device_utils.DEVICE, "cpu")
        finally:
            device_utils.DEVICE = original_device

    def test_enable_mps_rtmcc_accuracy_compatibility_patches_head_loss(self):
        from mmpose.models.heads.coord_cls_heads.rtmcc_head import RTMCCHead

        original_loss = RTMCCHead.loss
        try:
            train_module.enable_mps_rtmcc_accuracy_compatibility()
            self.assertIsNot(RTMCCHead.loss, original_loss)
        finally:
            RTMCCHead.loss = original_loss


if __name__ == "__main__":
    unittest.main()
