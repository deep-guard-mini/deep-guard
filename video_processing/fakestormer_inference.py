from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision import transforms

REPO_DIR = Path(__file__).resolve().parent / "FakeSTormer_repo"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from mmengine.config import Config
from models import MODELS, build_model
from .video_loader import extract_face_frames_with_metadata


class FakeSTormerVideoDetector:
    """Video-level deepfake detector backed by the trained FakeSTormer C23 model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ):
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.is_file():
            raise FileNotFoundError(
                f"FakeSTormer model checkpoint not found: {ckpt_file}"
            )

        cfg_file = (
            Path(config_path)
            if config_path
            else REPO_DIR / "configs" / "temporal" / "FakeSFormer_base_c23.yaml"
        )
        if not cfg_file.is_file():
            raise FileNotFoundError(
                f"FakeSTormer configuration file not found: {cfg_file}"
            )

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = ckpt_file
        self.config_path = cfg_file

        self.cfg = Config.fromfile(str(cfg_file))
        self.cfg.TRAIN = False
        self.cfg.FROM_FILE = False

        self.model = build_model(self.cfg.MODEL, MODELS)
        checkpoint = torch.load(
            str(ckpt_file), map_location=self.device, weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # Infer exact clip frame count from backbone time_embed (e.g. 5 - 1 = 4 frames)
        backbone = getattr(self.model, "backbone", None)
        if backbone is not None and hasattr(backbone, "time_embed"):
            has_temp_token = getattr(backbone, "temp_token", None) is not None
            self.num_frames = backbone.time_embed.size(1) - (1 if has_temp_token else 0)
        else:
            self.num_frames = 4

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def predict_video(self, video_path: str | Path, threshold: float = 0.5) -> dict:
        # Sample 4 clips of num_frames across the video and average predictions
        total_samples = self.num_frames * 4
        all_frames, faces_detected = extract_face_frames_with_metadata(
            video_path, num_frames=total_samples
        )

        clip_scores = []
        with torch.inference_mode():
            for i in range(0, total_samples, self.num_frames):
                clip_frames = all_frames[i : i + self.num_frames]
                tensors = torch.stack(
                    [self.transform(f) for f in clip_frames]
                )  # (T, C, H, W)
                video_tensor = (
                    tensors.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
                )  # (1, C, T, H, W)

                outputs = self.model(video_tensor)
                # Unpack classification logits or spatial-temporal heatmap
                if isinstance(outputs, (tuple, list)):
                    last = outputs[-1]
                    if isinstance(last, dict):
                        cls_out = next(
                            (
                                v
                                for k, v in last.items()
                                if "cls" in k.lower()
                                or "pred" in k.lower()
                                or "out" in k.lower()
                            ),
                            next(iter(last.values())),
                        )
                    else:
                        cls_out = (
                            last
                            if hasattr(last, "ndim") and last.ndim == 2
                            else outputs[0]
                        )
                elif isinstance(outputs, dict):
                    cls_out = outputs.get("cls_pred", next(iter(outputs.values())))
                else:
                    cls_out = outputs

                if cls_out.ndim == 2 and cls_out.shape[-1] == 2:
                    prob = float(torch.softmax(cls_out, dim=-1)[0, 1].item())
                elif cls_out.ndim == 2 and cls_out.shape[-1] == 1:
                    prob = float(torch.sigmoid(cls_out[0, 0]).item())
                else:
                    prob = float(torch.sigmoid(cls_out).mean().item())
                clip_scores.append(prob)

        fake_score = sum(clip_scores) / len(clip_scores)
        fake_prob = round(fake_score, 4)
        real_prob = round(1.0 - fake_score, 4)
        is_fake = fake_score >= threshold
        label = "FAKE" if is_fake else "REAL"
        return {
            "prediction": label,
            "label": label,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "frames_analyzed": len(all_frames),
            "faces_detected": faces_detected,
        }
