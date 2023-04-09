import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class cokes():
    def __init__(self, weights, source, conf, imgsz=640, device='0', iou_thres=0.45):
        """
        The function loads the model, sets the device, and sets the image size
        
        :param weights: the path to the weights file
        :param source: the path to the video file
        :param conf: the confidence threshold for the bounding boxes
        :param imgsz: The size of the image to be processed, defaults to 640 (optional)
        :param device: the GPU device to use, defaults to 0 (optional)
        :param iou_thres: The IoU threshold for non-maximum suppression
        """
        self.__weights = weights
        self.conf = conf
        self._source = source
        self.__imgsz = imgsz
        self.iou_thres = iou_thres

        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.__weights, map_location=self.device)  # load FP32 model
        self.__stride = int(self.model.stride.max())  # model stride
        self.__imgsz = check_img_size(imgsz, s=self.__stride)  # check img_size
        
        trace = True
        if trace:
            self.model = TracedModel(self.model, self.device, self.__imgsz)

        if self.half:
            self.model.half()  # to FP16

    def detect(self):
        dataset = LoadImages(self._source, img_size=self.__imgsz, stride=self.__stride)

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.__imgsz, self.__imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf, self.iou_thres, classes=None, agnostic=False)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        return im0


if __name__ == '__main__':
    weights_path = 'runs/train/exp/weights/best.pt'
    img_path = './drinks-1/test/images/17_jpg.rf.1c5c6ad84cb7dea7c406c86e0c12321c.jpg'
    pred = cokes(weights=weights_path, source=img_path, conf=0.1)

    with torch.no_grad():
        ODimages = pred.detect()
        
    start_time = time.time()
    ODimages = pred.detect()
    cv2.imshow(f'time: {time.time()-start_time}', ODimages)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 