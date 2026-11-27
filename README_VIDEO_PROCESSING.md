# DeepGuard — Video Processing Integration (FakeSTormer C23)

See `video_processing/README.md` for full architecture, interface, and checkpoint setup details.

## Quick Summary

- **Package:** `video_processing/` (`predict_video`, `predict_video_file`, `get_video_detector`)
- **Model:** FakeSTormer (`TopDownDetector` + `TimeViT` backbone), config `video_processing/FakeSTormer_repo/configs/temporal/FakeSFormer_base_c23.yaml`
- **Backend integration:** `backend/services/video_service.py` calls `video_processing.predictor.predict_video()`
- **Dependencies:** Install `backend/requirements.txt` and `requirements-video-processing.txt`
- **Checkpoint (~943 MB, excluded from Git):** Place at `models/fakestormer_c23_best.pth` (or `models/TopDownDetector_C23_ViTBase224_ST_hm100_tempLOC0.2_4SBI_SAM_mp0.01_temp2_0vidAug_0.35_temp_normlms_normREAL_model_best.pth`), or set `FAKESTORMER_MODEL_PATH` / `FAKESTORMER_WEIGHTS_URL`.
