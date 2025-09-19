# [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)
This project integrates the YOLOv10 object detection, CCD camera control, and GSW lighting control systems, enabling real-time object detection and precise positioning-based lighting functions.

Official PyTorch implementation of **YOLOv10**.

[YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458).\
Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, and Guiguang Ding\
[![arXiv](https://img.shields.io/badge/arXiv-2405.14458-b31b1b.svg)](https://arxiv.org/abs/2405.14458) <a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov10-object-detection-on-custom-dataset.ipynb#scrollTo=SaKTSzSWnG7s"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kadirnar/Yolov10)   [![Transformers.js Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers.js-blue)](https://huggingface.co/spaces/Xenova/yolov10-web)

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Over the past years, YOLOs have emerged as the predominant paradigm in the field of real-time object detection owing to their effective balance between computational cost and detection performance. Researchers have explored the architectural designs, optimization objectives, data augmentation strategies, and others for YOLOs, achieving notable progress. However, the reliance on the non-maximum suppression (NMS) for post-processing hampers the end-to-end deployment of YOLOs and adversely impacts the inference latency. Besides, the design of various components in YOLOs lacks the comprehensive and thorough inspection, resulting in noticeable computational redundancy and limiting the model's capability. It renders the suboptimal efficiency, along with considerable potential for performance improvements. In this work, we aim to further advance the performance-efficiency boundary of YOLOs from both the post-processing and the model architecture. To this end, we first present the consistent dual assignments for NMS-free training of YOLOs, which brings the competitive performance and low inference latency simultaneously. Moreover, we introduce the holistic efficiency-accuracy driven model design strategy for YOLOs. We comprehensively optimize various components of YOLOs from both the efficiency and accuracy perspectives, which greatly reduces the computational overhead and enhances the capability. The outcome of our effort is a new generation of YOLO series for real-time end-to-end object detection, dubbed YOLOv10. Extensive experiments show that YOLOv10 achieves the state-of-the-art performance and efficiency across various model scales. For example, our YOLOv10-S is 1.8$\times$ faster than RT-DETR-R18 under the similar AP on COCO, meanwhile enjoying 2.8$\times$ smaller number of parameters and FLOPs. Compared with YOLOv9-C, YOLOv10-B has 46\% less latency and 25\% fewer parameters for the same performance.
</details>

## Installation
`conda` virtual environment is recommended. 
```
conda create -n yolov10 python=3.9
conda activate yolov10
pip install -r requirements.txt
pip install -e .
```
## Demo
```
wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt
python app.py
# Please visit http://127.0.0.1:7860
```

## Validation
[`yolov10n.pt`](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt)  [`yolov10s.pt`](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt)  [`yolov10m.pt`](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt)  [`yolov10b.pt`](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt)  [`yolov10l.pt`](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt)  [`yolov10x.pt`](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt)  
```
yolo val model=yolov10n/s/m/b/l/x.pt data=coco.yaml batch=256
```

## Training 
```
yolo detect train data=coco.yaml model=yolov10n/s/m/b/l/x.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
```

## Prediction
```
yolo predict model=yolov10n/s/m/b/l/x.pt
```

## Export
```
# End-to-End ONNX
yolo export model=yolov10n/s/m/b/l/x.pt format=onnx opset=13 simplify
# Predict with ONNX
yolo predict model=yolov10n/s/m/b/l/x.onnx

# End-to-End TensorRT
yolo export model=yolov10n/s/m/b/l/x.pt format=engine half=True simplify opset=13 workspace=16
# Or
trtexec --onnx=yolov10n/s/m/b/l/x.onnx --saveEngine=yolov10n/s/m/b/l/x.engine --fp16
# Predict with TensorRT
yolo predict model=yolov10n/s/m/b/l/x.engine
```

## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR).

Thanks for the great implementations! 

## Citation

If our code or models help your work, please cite our paper:
```BibTeX
@misc{wang2024yolov10,
      title={YOLOv10: Real-Time End-to-End Object Detection}, 
      author={Ao Wang and Hui Chen and Lihao Liu and Kai Chen and Zijia Lin and Jungong Han and Guiguang Ding},
      year={2024},
      eprint={2405.14458},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
