# MEF-VR

## Install

Linux or WSL2, Python 3.10, CUDA GPU with at least 20 GiB free VRAM, and at least 40 GiB free disk space plus video scratch space.

```bash
python -m pip install -r requirements.txt
```

## Run

Run the complete detector on all 97 evaluation videos:

```bash
python run.py
```

The default output is `outputs/run`. Use a new output directory for each experiment.

| Operation | Command |
|---|---|
| Complete evaluation and input interventions | `python run.py --ablations --output outputs/ablations` |
| One new video | `python run.py --video input.mp4 --output outputs/video` |
| Short execution check | `python run.py --ids H12 --output outputs/check` |
| Recalculate archived metrics without GPU inference | `python run.py --metrics --output outputs/tables` |
| Resume the same evaluation | `python run.py --resume` |
| Extract final training features | `python run.py --split train --keep-features --output outputs/train_features` |
| Full Head training entry | `python run.py --train full --features outputs/train_features --output outputs/full_training` |
| Context Head training entry | `python run.py --train context --features outputs/train_features --output outputs/context_training` |
| Evaluate synthetic pixel/frame outputs | `python run.py --localization outputs/dev --output outputs/localization` |

`--split development` evaluates the development set. Training entries are optional; `python run.py` uses the included trained weights for detection. Training accepts `--arm`, `--seed`, `--epochs`, and `--check-only`. New training outputs never overwrite the supplied checkpoints.

## Files and outputs

`run.py` is the only public entry point. `mef_vr/` contains its required implementation and training code. `models/` contains all detector weights; `configs/` contains fixed settings and integrity bindings.

`data/` contains 300 final-stage training videos, 200 development videos, 97 real evaluation videos, input scopes and labels. Earlier backbone-training histories are represented by the selected checkpoints, not by every historical training dataset. Common-81 is a subset of Real-97. `results.json` consolidates the archived experiment records; it is not used as input to raw-video detection.

Each video produces `prediction.json`, native-clock frame scores and region candidates. Batch results are written to `all_results.json` with calculated metrics. `--ablations` applies fixed-weight input/rule interventions. The supplied `results.json` contains only final Real-97 results, component interventions, API comparisons and other baselines. Missing predictions remain failures, not Clean labels. The archived complete Real-97 counts are TP=36, FN=9, FP=10, TN=42. 

All detection modules run internally, including temporal, appearance, motion, regional, fusion, interpolation, dark-flash and endpoint checks. No API key or paired Clean video is needed at inference. Evaluation uses the archived frame intervals and approved viewport; `--video` processes the entire supplied file.

## Third-party notices

R3D-18 and RAFT use TorchVision implementations. DINOv2-small is attributed to Meta and its downloaded model card specifies Apache-2.0. Third-party libraries and assets retain their original licences. Source videos and scene assets retain their owners' rights; this package does not grant unrestricted public redistribution rights.

The included RIFE source carries the following notice:

MIT License

Copyright (c) Megvii Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
