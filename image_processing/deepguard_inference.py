
import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50


class DeepGuardImageDetector:

    def __init__(self, checkpoint_path, device="cpu"):

        self.device = torch.device(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        self.class_mapping = checkpoint["class_mapping"]
        self.index_to_label = {
            value: key
            for key, value in self.class_mapping.items()
        }

        image_size = checkpoint["image_size"]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=checkpoint["imagenet_mean"],
                std=checkpoint["imagenet_std"]
            )
        ])

        self.model = resnet50(weights=None)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image):

        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))

        if not isinstance(image, Image.Image):
            raise TypeError(
                "Expected PIL Image or image bytes"
            )

        image = image.convert("RGB")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        logits = self.model(tensor)

        probabilities = torch.softmax(
            logits, dim=1
        )[0]

        predicted_index = int(
            probabilities.argmax().item()
        )

        predicted_label = self.index_to_label[
            predicted_index
        ]

        return {
            "prediction": predicted_label,
            "confidence": round(
                float(probabilities[predicted_index].item()),
                4
            ),
            "probabilities": {
                "REAL": round(
                    float(probabilities[
                        self.class_mapping["REAL"]
                    ].item()),
                    4
                ),
                "FAKE": round(
                    float(probabilities[
                        self.class_mapping["FAKE"]
                    ].item()),
                    4
                )
            }
        }
