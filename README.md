# Object detection

YOLOv1, YOLOv2 and YOLOv3 written from scratch in PyTorch and trained to detect
bees at a hive entrance. Each generation is a separate pair of files, and every
version only adds what its paper added, so the diff between two generations is
the contribution of the later paper:

| Generation | Model | Loss | Paper |
| --- | --- | --- | --- |
| v1 | `yolo.py` | `yolo_loss.py` | [arXiv:1506.02640](https://arxiv.org/abs/1506.02640) |
| v2 | `yolov2.py` | `yolov2_loss.py` | [arXiv:1612.08242](https://arxiv.org/abs/1612.08242) |
| v3 | `yolov3.py` | `yolov3_loss.py` | [arXiv:1804.02767](https://arxiv.org/abs/1804.02767) |

`yolov3.py` reuses v2's `conv_bn` and `grid_to_boxes`, and `YOLOv3Loss`
subclasses `YOLOv2Loss`, overriding only the objectness and class terms. The
module docstrings list the changes from the previous generation.

## Results

Single class (bee), mAP@0.5 on a held-out 10% validation split. v3 reaches
**0.873** after 38 epochs, starting from the pretrained Darknet-53 backbone.
v1 on the other hand genuinely cannot do this task. Its
7x7 grid holds at most one object per cell, so on frames with twenty-odd bees
packed along a hive entrance most of the ground truth is discarded at encoding
time (`BeeDataset.encode_target` drops boxes landing in an occupied cell). The
multi-scale anchor grids in v2/v3 are what fix it.

The Darknet-53 classification pretraining reaches 0.974 accuracy on bee vs.
background crops, and the v3 detector above starts from that backbone.

## Setup

Python 3.10 or newer is required (the code uses `X | Y` annotations);
developed on 3.12.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Download the bee dataset from [Dataset Ninja](https://datasetninja.com/bee-image)
and unpack it so the annotation and image directories sit side by side:

```
Bees/ds/img/<name>.jpeg
Bees/ds/ann/<name>.jpeg.json
```

3044 images in Supervisely JSON format, annotated with axis-aligned rectangles.
Resolutions are mixed: 4032x3024 (1463), 1280x720 (1053), 720x1280 (432) and
3024x4032 (96). Images are stretched to a square network input rather than
letterboxed, consistently at training and inference time.

## Usage

Pretrain a backbone on bee/background crops, then train the detector on top of
it:

```bash
python bee_pretrain.py  --model v3 --epochs 10
python bee_detection.py --model v3 --backbone_weights runs/pretrain_v3/backbone.pt
```

Training from scratch works too, just drop `--backbone_weights`. v1 has no
pretraining stage.

```bash
python bee_detection.py --model v1                      # trains at 448, 7x7 grid
python bee_detection.py --model v2                      # 416, 5 k-means anchors
python bee_detection.py --model v3                      # 416, 9 anchors over 3 scales
```

Evaluate a checkpoint and draw predictions:

```bash
python bee_detection.py --model v3 --eval_only --weights runs/bee_yolov3/best.pt
python bee_show.py --weights runs/bee_yolov3/best.pt --num 12   # Generates 12 images
```

`bee_show.py` writes one `<name>_pred.jpg` per image into `<checkpoint dir>/vis`,
with ground truth in green and predictions in red. Output keeps the source
resolution, and the box and label sizes scale with the image so the overlay
reads the same on a 720p frame and on a 12-megapixel photo.

Useful flags while iterating:

- `--overfit N` trains and validates on the same N images with augmentation
  off. A working pipeline reaches mAP near 1.0 within a few hundred steps, so
  this is the fastest way to tell a broken pipeline from a hard dataset.
- `--max_steps N` for smoke tests, `--eval_batches N` to cap evaluation cost.
- `--amp` for mixed precision on CUDA, `--workers 0` to load in the main
  process when debugging.
- `--resume` continues from `last.pt`.

Checkpoints land in `--save_dir` (default `runs/bee_yolo<model>`): `last.pt`
every `--save_every` steps with optimizer and scheduler state, and `best.pt`
whenever validation mAP improves. `best.pt` holds weights only, so resume from
`last.pt`.

## Layout

| File | Contents |
| --- | --- |
| `paper.py` | Constants fixed by the YOLOv1 paper: grid size, input size, loss weights, epochs |
| `yolo.py`, `yolo_loss.py` | YOLOv1: 24 conv + 2 FC layers, sum-squared-error loss |
| `yolov2.py`, `yolov2_loss.py` | YOLOv2: Darknet-19, passthrough, anchor-space regression |
| `yolov3.py`, `yolov3_loss.py` | YOLOv3: Darknet-53, FPN-style three-scale head, BCE objectness |
| `bboxes_utils.py` | IoU, NMS, coordinate conversion, k-means anchor clustering |
| `bee_dataset.py` | Supervisely JSON to grid targets and normalised boxes |
| `bee_crops.py` | Bee/background crops for classification pretraining |
| `bee_detection.py` | Detector training, evaluation and mAP |
| `bee_pretrain.py` | Backbone classification pretraining |
| `bee_show.py` | Draws predictions next to ground truth |

## Implementation notes

- Anchors are k-means clusters (distance `1 - IoU`, as in the v2 paper) of the
  training boxes, computed at startup and stored as a model buffer, so a
  checkpoint carries the priors it was trained with.
- The dataset has no class labels, so everything runs with `--num_classes 1`
  and the class term of each loss is inert. The multi-class paths are
  implemented but untested.
- Optimisation deviates from the papers: AdamW with linear warmup and cosine
  decay, since the papers' hand-tuned SGD schedules have no AdamW equivalent.
  The paper values that do carry over live in `paper.py`.
- mAP is computed with VOC all-point interpolation on normalised coordinates.
  Because images are stretched to a square, that space is horizontally
  compressed by the original aspect ratio, so the number is consistent across
  runs here but not identical to a pixel-space evaluator.
- Not implemented: multi-scale training and the 448 high-resolution classifier
  fine-tune from v2, and the "things we tried that didn't work" from v3.

## License

MIT, see [LICENSE](LICENSE).
