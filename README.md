# 🌊 OceanWatch: underwater marine litter detection

Comparing **YOLO26-l** (CNN) and **RT-DETR-l** (transformer) for detecting marine litter in underwater ROV images from the [TrashCan 1.0](https://conservancy.umn.edu/handle/11299/214865) dataset.

**Live app:** _add your Streamlit link here_

## Results (test set: 1,081 images from 57 unseen videos)

| | YOLO26-l | RT-DETR-l |
|---|---|---|
| Litter (trash) AP50 | 0.560 | **0.655** |
| mAP50 (all classes) | 0.498 | 0.500 |
| mAP50-95 | 0.311 | **0.346** |
| Precision / Recall | 0.639 / 0.494 | 0.675 / 0.462 |
| Speed (Tesla T4, 1 image) | **25.8 ms (38.8 FPS)** | 79.7 ms (12.6 FPS) |

RT-DETR-l finds litter more accurately; YOLO26-l is about 3× faster.

## Method
- 7,212 TrashCan images, 22 classes merged into **trash / bio / rov**.
- **Video-level split** (70/15/15): the official split shares frames from 113 of its 115 test videos with training, so we keep whole videos together.
- Both models COCO-pretrained, trained 40 epochs with identical image size (640), batch (16), seed and data (Ultralytics 8.4.177).

## Repository
| Path | Contents |
|---|---|
| `app.py` | Streamlit app: upload an image → both models' detections, confidence scores, per-image accuracy for labelled test images, overall results |
| `notebook/OceanWatch_Final.ipynb` | Full pipeline (Google Colab): data preparation, training, testing, experiments |
| `samples/` | Test images with their ground-truth labels (YOLO format) |
| `results/` | Test-set results and the enhancement experiment |
| Release `v1.0` | Trained model weights (`yolo26l_3cls.pt`, `rtdetr_l_3cls.pt`), downloaded automatically by the app |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## References
- Hong, J., Fulton, M., & Sattar, J. (2020). TrashCan 1.0: An Instance-Segmentation Labeled Dataset of Trash Observations. University of Minnesota. https://doi.org/10.13020/g1gx-y834
- Zhao, Y. et al. (2024). DETRs Beat YOLOs on Real-time Object Detection. CVPR.
- Ultralytics YOLO26 and RT-DETR: https://docs.ultralytics.com

*MAAI901 Deep Learning for Visual Data, University of Wollongong in Dubai.*
