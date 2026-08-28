from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LocalAiContainerConfigTests(unittest.TestCase):
    def test_compose_uses_real_read_only_model_mounts_and_local_port(self):
        compose = (ROOT / "compose.local-ai.yaml").read_text()

        self.assertIn("127.0.0.1:8000:8000", compose)
        self.assertIn("CARD_HAS_MODEL_HOST", compose)
        self.assertIn("INFANT_HEAD_TOP_CHECKPOINT_HOST", compose)
        self.assertIn("/models/card.pt:ro", compose)
        self.assertIn("/models/head_top.pth:ro", compose)
        self.assertNotIn("HEIGHT_MEASUREMENT_PROVIDER=mock", compose)

    def test_image_installs_full_cpu_mmcv_and_does_not_copy_weights(self):
        dockerfile = (ROOT / "Dockerfile.local-cpu").read_text()
        dockerignore = (ROOT / ".dockerignore").read_text()

        self.assertIn("mmcv==2.1.0", dockerfile)
        self.assertIn(
            "download.openmmlab.com/mmcv/dist/cpu/torch2.1.0",
            dockerfile,
        )
        self.assertIn("--no-build-isolation chumpy==0.70", dockerfile)
        self.assertIn("torch|torchvision|mmcv|chumpy", dockerfile)
        self.assertNotIn("COPY work_dirs", dockerfile)
        self.assertNotIn("COPY app/models", dockerfile)
        self.assertIn("*.pt", dockerignore)
        self.assertIn("*.pth", dockerignore)


if __name__ == "__main__":
    unittest.main()
