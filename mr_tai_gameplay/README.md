# Quickstart

```bash
cd mr_tai_gameplay
bash setup.sh

# Inference on a single clip
python -m src.pipeline_infer path/to/clip.mp4 --out out/pred.json --device mps # on Mac

# Training (optional, on your weak labels)
python -m train.train_baseline train/dataset_index.json --device mps
# then use the checkpoint for inference
python -m src.pipeline_infer path/to/clip.mp4 --ckpt out_ckpt/r3d18_ep10.pt --device mps