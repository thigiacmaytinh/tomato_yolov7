## Install package and library
- Download Dataset from Robotflow: Đã download trong folder drinks-1
- Install Python 3.7.3 AMD64
- Install Enviroment: 
```
pip install -r requirements.txt
```
- Install CUDA 11.3 and CuDNN 8.2.1.32

- Install Pytorch

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
- Install Visual Studio Code

- Download file pretrain
```
https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```
## Training:

Training bằng file notebook yolov7_training.ipynb


## Ghi chú

### Chạy file client.py
- Leak mem -> comment dòng x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf trong file ultis/general.py (dòng 648->652)

### Điều chỉnh tham số truyền vào class coke trong file client.py

param weights: the path to the weights file
param source: the path to the video file
param conf: the confidence threshold for the bounding boxes
param imgsz: The size of the image to be processed, defaults to 640 (optional)
param device: the GPU device to use, defaults to 0 (optional)
param iou_thres: The IoU threshold for non-maximum suppression

### Ảnh sau khi Detect sẽ được lưu trong run/detect/exp