# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.to_txt02 import creat_txt
from utils.txt2xml import save_xml_file


note_classes = ['Sharp', 'Flat', 'Natural','TimeSig', 'Rests1', 'Rests2', 'Rests4', 'Rests8', 'Rests16','Rests32', 'Rests64', 'Rests128','Gclef', 'Fclef', 'Cclef',
                'High_Gclef', 'DHigh_Gclef', 'Lower_Gclef', 'DLower_Gclef', 'DGclef','Soprano_Cclef', 'M_Soprano_Cclef', 'Tensor_Cclef', 'High_Fclef', 'DHigh_Fclef', 'Lower_Fclef','DLower_Fclef', 'Up_Fclef', 'Tensor_Fclef',
                'D_S', 'A_S', 'E_S', 'B_S', 'F_S', 'C_S', 'B_F', 'E_F', 'A_F','D_F', 'G_F', 'C_F',
                'Speed_4=80', 'Speed_4=160', 'Speed_4=123', 'Speed_2=80', 'Speed_4=95', 'Speed_4=57', 'Speed_4=123','Speed_4=50', 'Speed_4=120', 'Speed_4=100', 'Speed_4=110', 'Speed_4=105', 'Speed_4=150', 'Speed_4=140','Speed_4=84', 'Speed_4=60', 'Speed_4=30',
                '1_-10', '1_-9', '1_-8', '1_-7', '1_-6', '1_-5', '1_-4', '1_-3', '1_-2', '1_-1', '1_0', '1_1', '1_2', '1_3','1_4', '1_5', '1_6', '1_7', '1_8', '1_9', '1_10', '1_11', '1_12', '1_13', '1_14', '1_15', '1_16', '1_17','1_18', '1_19', '1_20',
                '2_-10', '2_-9', '2_-8', '2_-7', '2_-6', '2_-5', '2_-4', '2_-3', '2_-2', '2_-1', '2_0', '2_1', '2_2', '2_3','2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '2_10', '2_11', '2_12', '2_13', '2_14', '2_15', '2_16', '2_17','2_18', '2_19', '2_20',
                '4_-10', '4_-9', '4_-8', '4_-7', '4_-6', '4_-5', '4_-4', '4_-3', '4_-2', '4_-1', '4_0', '4_1', '4_2', '4_3','4_4', '4_5', '4_6', '4_7', '4_8', '4_9', '4_10', '4_11', '4_12', '4_13', '4_14', '4_15', '4_16', '4_17','4_18', '4_19', '4_20',
                '8_-10', '8_-9', '8_-8', '8_-7', '8_-6', '8_-5', '8_-4', '8_-3', '8_-2', '8_-1', '8_0', '8_1', '8_2', '8_3','8_4', '8_5', '8_6', '8_7', '8_8', '8_9', '8_10', '8_11', '8_12', '8_13', '8_14', '8_15', '8_16', '8_17','8_18', '8_19', '8_20',
                '16_-10', '16_-9', '16_-8', '16_-7', '16_-6', '16_-5', '16_-4', '16_-3', '16_-2', '16_-1', '16_0', '16_1','16_2', '16_3', '16_4', '16_5', '16_6', '16_7', '16_8', '16_9', '16_10', '16_11', '16_12', '16_13', '16_14','16_15', '16_16', '16_17', '16_18', '16_19', '16_20',
                '32_-10', '32_-9', '32_-8', '32_-7', '32_-6', '32_-5', '32_-4', '32_-3', '32_-2', '32_-1', '32_0', '32_1','32_2', '32_3', '32_4', '32_5', '32_6', '32_7', '32_8', '32_9', '32_10', '32_11', '32_12', '32_13', '32_14','32_15', '32_16', '32_17', '32_18', '32_19', '32_20',
                '64_-10', '64_-9', '64_-8', '64_-7', '64_-6', '64_-5', '64_-4', '64_-3', '64_-2', '64_-1', '64_0', '64_1','64_2', '64_3', '64_4', '64_5', '64_6', '64_7', '64_8', '64_9', '64_10', '64_11', '64_12', '64_13', '64_14','64_15', '64_16', '64_17', '64_18', '64_19', '64_20',
                '128_-10', '128_-9', '128_-8', '128_-7', '128_-6', '128_-5', '128_-4', '128_-3', '128_-2', '128_-1', '128_0','128_1', '128_2', '128_3', '128_4', '128_5', '128_6', '128_7', '128_8', '128_9', '128_10', '128_11','128_12', '128_13', '128_14', '128_15', '128_16', '128_17', '128_18', '128_19', '128_20',
                '1_-7_0', '1_-5_2', '1_-4_-1', '1_-4_0', '1_-4_3', '1_-3_4', '1_-2_0', '1_-2_3', '1_-2_5', '1_-1_3','1_-1_6', '1_-1_10', '1_0_3', '1_0_4', '1_0_5', '1_0_6', '1_0_7', '1_1_2', '1_1_3', '1_1_4', '1_1_5','1_1_8', '1_2_4', '1_2_5', '1_2_9', '1_3_4', '1_3_5', '1_3_7', '1_3_10','1_4_6', '1_4_10', '1_5_7', '1_5_8', '1_5_10', '1_6_7', '1_6_8', '1_6_9', '1_6_10', '1_6_11', '1_6_13','1_7_9', '1_7_10', '1_8_10', '1_8_11', '1_9_10', '1_10_12', '1_12_15', '1_12_16',
                '2_-9_-2', '2_-8_-1', '2_-7_-4', '2_-7_-3', '2_-7_-2', '2_-7_0', '2_-6_-2', '2_-6_1', '2_-5_-2', '2_-5_-1','2_-5_0', '2_-5_2', '2_-4_-2', '2_-4_-1', '2_-4_0', '2_-4_2', '2_-4_3', '2_-3_-1', '2_-3_1', '2_-3_2','2_-3_3', '2_-3_4', '2_-2_0', '2_-2_2', '2_-2_3', '2_-2_5','2_-2_10', '2_-1_1', '2_-1_2', '2_-1_3', '2_-1_4', '2_-1_5', '2_-1_6', '2_-1_8', '2_-1_9', '2_0_2', '2_0_3','2_0_4', '2_0_5', '2_0_6', '2_0_7', '2_0_9', '2_1_2', '2_1_3', '2_1_4', '2_1_5', '2_1_6', '2_1_8', '2_2_4','2_2_5', '2_2_6', '2_2_7', '2_2_8', '2_2_9', '2_2_11','2_3_5', '2_3_6', '2_3_7', '2_3_8', '2_3_9', '2_3_10', '2_4_6', '2_4_7', '2_4_8', '2_4_9', '2_4_11', '2_5_7','2_5_8', '2_5_9', '2_5_10', '2_5_12', '2_6_7', '2_6_8', '2_6_9', '2_6_10', '2_6_11', '2_6_13', '2_7_8','2_7_9', '2_7_10', '2_7_12', '2_7_14', '2_8_10', '2_8_11', '2_8_13', '2_9_11', '2_9_13', '2_10_12','2_11_12', '2_12_14', '2_13_14', '2_13_15',
                '4_-9_-4', '4_-9_-2', '4_-9_-1', '4_-8_-1', '4_-7_-2', '4_-7_-1', '4_-7_0', '4_-6_-2', '4_-6_1', '4_-5_-3','4_-5_-1', '4_-5_0', '4_-5_1', '4_-5_2', '4_-5_3', '4_-5_4', '4_-4_-2', '4_-4_-1', '4_-4_0', '4_-4_1','4_-4_3', '4_-4_5', '4_-3_-2', '4_-3_-1', '4_-3_0', '4_-3_1','4_-3_2', '4_-3_4', '4_-3_5', '4_-3_6', '4_-2_-1', '4_-2_0', '4_-2_1', '4_-2_2', '4_-2_3', '4_-2_4','4_-2_5', '4_-2_7', '4_-1_1', '4_-1_3', '4_-1_4', '4_-1_5', '4_-1_6', '4_-1_7', '4_0_2', '4_0_3', '4_0_4','4_0_5', '4_0_7', '4_1_2', '4_1_3', '4_1_4', '4_1_5', '4_1_6', '4_1_7','4_1_8', '4_1_10', '4_2_4', '4_2_5', '4_2_6', '4_2_7', '4_2_8', '4_2_9', '4_3_4', '4_3_5', '4_3_6', '4_3_7','4_3_8', '4_3_9', '4_3_10', '4_4_6', '4_4_7', '4_4_8', '4_4_9', '4_4_10', '4_4_11', '4_5_6', '4_5_7','4_5_8', '4_5_9', '4_5_10', '4_5_12', '4_6_7', '4_6_8', '4_6_9', '4_6_10','4_6_11', '4_6_12', '4_6_13', '4_7_9', '4_7_10', '4_7_11', '4_7_12', '4_7_13', '4_7_14', '4_8_9', '4_8_10','4_8_11', '4_8_12', '4_8_13', '4_8_14', '4_8_15', '4_9_10', '4_9_11', '4_9_12', '4_9_13', '4_9_14', '4_9_16','4_10_12', '4_10_13', '4_10_14', '4_10_15', '4_10_17', '4_11_13', '4_12_14', '4_12_15', '4_13_15',
                '8_-9_-2', '8_-9_-1', '8_-8_-1', '8_-7_0', '8_-7_1', '8_-7_2', '8_-6_-2', '8_-6_1', '8_-6_2', '8_-6_3','8_-5_-1', '8_-5_0', '8_-5_2', '8_-5_3', '8_-5_9', '8_-4_-2', '8_-4_-1', '8_-4_0', '8_-4_1', '8_-4_3','8_-4_4', '8_-3_-2', '8_-3_-1', '8_-3_0', '8_-3_1', '8_-3_2', '8_-3_3', '8_-3_4', '8_-2_-1', '8_-2_0','8_-2_1','8_-2_2', '8_-2_3', '8_-2_4', '8_-2_5', '8_-2_7', '8_-2_8', '8_-2_9', '8_-2_10', '8_-1_1', '8_-1_2','8_-1_3', '8_-1_4', '8_-1_5', '8_-1_6', '8_-1_7', '8_-1_8', '8_-1_9', '8_0_2', '8_0_3', '8_0_4', '8_0_5','8_0_7', '8_0_8', '8_1_2', '8_1_3', '8_1_4', '8_1_5', '8_1_6', '8_1_7', '8_1_8', '8_1_9', '8_2_3', '8_2_3','8_2_4','8_2_4', '8_2_5', '8_2_6', '8_2_7', '8_2_8', '8_2_9', '8_3_4', '8_3_5', '8_3_6', '8_3_7', '8_3_8', '8_3_9','8_3_10', '8_4_6', '8_4_7', '8_4_8', '8_4_9', '8_4_11', '8_5_7', '8_5_8', '8_5_9', '8_5_10', '8_5_12','8_6_7', '8_6_8', '8_6_9', '8_6_10', '8_6_11', '8_6_12', '8_6_13', '8_7_8', '8_7_9', '8_7_10', '8_7_11','8_7_12','8_7_13', '8_7_14', '8_8_10', '8_8_11', '8_8_12', '8_8_13', '8_8_14', '8_8_15', '8_9_10', '8_9_11', '8_9_12','8_9_13', '8_9_14', '8_9_15', '8_9_16', '8_10_12', '8_10_14', '8_10_17', '8_11_13', '8_11_14', '8_11_18','8_12_14', '8_13_15',
                '16_-7_2', '16_-6_1', '16_-5_0', '16_-5_2', '16_-4_3', '16_-3_-1', '16_-3_1', '16_-3_4', '16_-2_0','16_-2_1', '16_-2_2', '16_-2_5', '16_-1_1', '16_-1_2', '16_-1_3', '16_-1_4', '16_-1_5', '16_-1_6', '16_0_2','16_0_5', '16_0_7', '16_1_2', '16_1_3', '16_1_4', '16_1_5', '16_1_6', '16_1_7', '16_1_8', '16_2_4', '16_2_4','16_2_5', '16_2_6', '16_2_7', '16_2_9', '16_3_5', '16_3_6', '16_3_7', '16_3_8', '16_3_10', '16_3_13','16_4_5', '16_4_6', '16_4_7', '16_4_8', '16_4_9', '16_4_10', '16_4_11', '16_5_7', '16_5_8', '16_5_9','16_5_10', '16_5_12', '16_6_7', '16_6_8', '16_6_9', '16_6_10', '16_6_11', '16_6_13', '16_7_9', '16_7_10','16_7_12','16_7_14', '16_8_10', '16_8_11', '16_8_13', '16_8_15', '16_9_11', '16_9_12', '16_9_14', '16_9_16','16_10_12', '16_10_13', '16_10_15', '16_10_17', '16_11_13', '16_12_14',
                '32_-3_0', '32_-3_1', '32_-3_2', '32_-3_6', '32_0_6', '32_2_6', '32_4_6', '32_5_7', '32_7_9', '32_8_10']


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    colors = []
    for label in names:
        if label == 'Sharp' or label == 'Flat' or label == 'Natural' or label == 'Cclef' or label == 'Gclef' or label == 'Fclef' or label == 'TimeSig':
            colors.append([255, 0, 0])
        elif label == 'Rests1' or label == 'Rests2' or label == 'Rests4' or label == 'Rests8' or label == 'Rests16' or label == 'Rests32' or label == 'Rests64' or label == 'Rests128':
            colors.append([0, 255, 255])
        elif label == 'Bass' or label == 'Guit' or label == 'Tpt' or label == 'Hn' or label == 'Tbn' or label == 'Tba' or label == 'Fl' or label == 'Ob' or label == 'Cl' or label == 'Bsn' or label == 'Vln' or label == 'Vla' or label == 'Vlc':
            colors.append([0, 255, 0])
        elif label == 'Pno' or label == 'Wh' or label == 'Trb' or label == 'Sax' or label == 'Hrp' or label == 'Cb' or label == 'Rec':
            colors.append([0, 0, 255])
        elif label == 's' or label == 'm' or label == 'p' or label == 'f' or label == 'mp' or label == 'mf' or label == 'sf' or label == 'ff' or label == 'fff' or label == 'fffff' or label == 'ffffff':
            colors.append([27, 171, 115])
        elif label == 'pp' or label == 'ppp' or label == 'pppp' or label == 'ppppp' or label == 'fp':
            colors.append([27, 171, 115])
        else:
            colors.append([255, 0, 255])
    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = []
                        line.append(names[int(cls)])
                        for i in xyxy:
                            line.append(str(int(i)))
                        with open(txt_path + '.txt', 'a') as f:
                            _str = ''
                            for i in line:
                                _str = _str + i + ','
                            _str = _str.replace(',', ' ').strip()
                            f.write(_str + '\n')
                        f.close()
                        # line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')  #{conf:.2f}
                        annotator.box_label(xyxy, label, color=colors[int(cls)])
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    image = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                    save_img = save_path.replace('jpg','pdf')
                    image.save(save_img)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

        label_path = save_dir / 'labels'
        mung_path = os.path.join(save_dir, "mung/")
        if not os.path.isdir(mung_path):
            os.mkdir(mung_path)

        csv_path = os.path.join(save_dir, "csv/")
        if not os.path.isdir(csv_path):
            os.mkdir(csv_path)

        list_img = [im0.shape[0], im0.shape[1], im0.shape[2], path]
        imgs_information = creat_txt(label_path, mung_path,list_img,note_classes)

        xml_dir = os.path.join(save_dir, "xml/")
        if not os.path.isdir(xml_dir):
            os.mkdir(xml_dir)
        save_xml_file(label_path, list_img, xml_dir,VOC2008=False)
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    return imgs_information

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/2009_v5/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'test2009/aa', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/voc2.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', default=True,action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=True,action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', default=False,help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    imgs_information = run(**vars(opt))
    return imgs_information


if __name__ == "__main__":
    opt = parse_opt()
    img_information = main(opt)