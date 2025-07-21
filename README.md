# Object Detection on Vehicle Interior Monitoring (SVIRO Dataset)

## YOLOv5l Model Setup, Preprocessing, and Training

The **YOLOv5l** model is sourced from the official [Ultralytics YOLOv5 GitHub repository](https://github.com/ultralytics/yolov5). It is pretrained on the COCO dataset, which contains over 200,000 labeled images spanning 80 diverse object categories. These pretrained weights provide a strong starting point, enabling effective fine-tuning on specialized datasets such as SVIRO for vehicle interior monitoring.

### Dataset Preparation and Preprocessing

- The SVIRO dataset’s original grayscale images are resized to **640×480** pixels using PIL to fit YOLOv5’s input size requirements.
- Bounding box annotations, initially in `[class_id, x_min, y_min, x_max, y_max]` format, are converted to the **YOLO format**: `[class_id, x_center, y_center, width, height]`, normalized by image dimensions.
- Class IDs are converted to zero-based indexing by subtracting 1.
- The dataset is organized into separate folders for training and testing splits, with labels and images stored accordingly.
- Non-relevant files (non-`.png` images and non-`.txt` labels) are cleaned up to ensure dataset integrity.

### Training Setup

- The YOLOv5 repository is cloned, and dependencies installed via `requirements.txt`.
- A custom YAML configuration file specifies dataset paths, class labels, and hyperparameters tailored to the SVIRO dataset.
- Training is initiated using the provided `train.py` script with parameters such as:
  ```bash
  python train.py --data data.yaml --epochs 56 --weights yolov5l.pt --cfg yolov5l.yaml --batch-size 8 --img 640 --image-weights
    ```

### Evaluation and Inference
- Model evaluation on the test split is performed using:
  ```bash
  python val.py --data data.yaml --weights runs/train/exp11/weights/best.pt --task test
    ```
- Example inference using the trained model:
    ```python
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp12/weights/best.pt')
result = model('yolo_dataset/images/test/grayscale_wholeImage/aclass_test_imageID_108_GT_3_4_6.png')
result.print()
result.show()
    ```

## Swin Transformer + RetinaNet Model Setup and Usage

On the other hand, the **Swin Transformer**, used alongside a RetinaNet detection head, was obtained from the [MMDetection model zoo](https://github.com/open-mmlab/mmdetection) by OpenMMLab. This model has also been pretrained on the COCO dataset, which provides strong feature representations that can be effectively fine-tuned for tasks related to vehicle interior monitoring, specifically tailored to the SVIRO dataset.

### Setup and Dataset Preparation

- The process begins by cloning the MMDetection repository and installing all necessary packages, including MMCV.
- The existing Swin Transformer RetinaNet configuration file was modified to reflect the specifics of the SVIRO dataset — including adjusting the number of classes, dataset format, and file paths.
- The SVIRO dataset annotations were converted from their original format into **COCO format JSON** files, suitable for MMDetection training:
  - Bounding boxes converted from `[class_id, x_min, y_min, x_max, y_max]` to COCO’s `[x_min, y_min, width, height]` format with absolute pixel values.
  - Category IDs start at 1, per COCO specification.
  - Images and annotations are organized into the required directory structure.

### Training

- Training is performed using MMDetection’s `tools/train.py` script with the customized Swin Transformer RetinaNet config:
  ```bash
  python mmdetection/tools/train.py mmdetection/configs/swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py --work-dir work_dirs/final1
    ```

### Inference, Visualization and Evaluation

- Initialize and run inference with the trained model:
  ```python
from mmdet.apis import init_detector, inference_detector
import mmcv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

config_file = 'mmdetection/configs/swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/final2/epoch_50.pth'
device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)
img = 'transformer_dataset/test/grayscale_wholeImage/aclass_test_imageID_108_GT_3_4_6.png'
result = inference_detector(model, img)
    ```

- Evaluate the trained model using MMDetection’s built-in COCO evaluation tools:
    ```bash
python mmdetection/tools/test.py configs/swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py work_dirs/final1/epoch_50.pth --cfg-options test_evaluator.metric="bbox" --out work_dirs/result.pkl
    ```