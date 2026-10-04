# DeepGaurd Architecture

## 1. Project Overview

DeepGaurd is an AI-based deepfake detection system for analyzing images
and videos and determining whether the media is real or manipulated.

The project is divided into independent modules so that each team member
can develop and test their component independently.

---

## 2. Project Structure

```text
deep-guard/
│
├── frontend/
│   └── Frontend application and UI
│
├── backend/
│   └── REST API and system integration
│
├── image-processing/
│   └── Image preprocessing and image analysis
│
├── video-processing/
│   └── Video processing and frame extraction
│
├── models/
│   └── ML/DL models, training and inference
│
├── tests/
│   └── Unit and integration tests
│
├── docs/
│   └── Project documentation
│
└── .github/
    └── workflows/
        └── CI/CD workflows