import argparse
import time
import os
import csv
import numpy as np
import pandas as pd
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
from utils.quantify import cal_movement_hist, cal_NNI
from utils.sort import *

def detect(save_img=False):
    start = time.time()
    COLORS = np.random.randint(0, 255, size=(200, 3),
                           dtype="uint8")


    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA------------------------------------------------------

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16


    # Set Dataloader
    vid_path, vid_writer = None, None
    for date in os.listdir(opt.source):
        date_path = os.path.join(opt.source, date)
        videos = {}
        for rpi in os.listdir(date_path):
            rpi_path = os.path.join(date_path, rpi) ## 20230818/aggressive/
            if os.path.isdir(rpi_path):
                for video in os.listdir(rpi_path):  
                    video_path = os.path.join(rpi_path, video)
                    if rpi in videos:
                        videos[rpi].append(video_path)
                    else:
                        videos[rpi] = [video_path]
        video_date = os.path.basename(date_path)
        for video in videos[rpi]:
            memory = {}
            tracked = {}
            movement_hist = []
            avg_movements = []
            avg_NNI = []
            std_movements = []
            std_NNI = []
            NNI = []
            move = {}
            dataset = LoadImages(video, img_size=imgsz, stride=stride)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            old_img_w = old_img_h = imgsz # 416 416 416
            old_img_b = 1
            tracker = Sort(max_age = 30, iou_threshold=0.5)
            t0 = time.time()
            vs = dataset.video_flag[0]
            frameIndex = 0
            
            for path, img, im0s, vid_cap in dataset:
                width1 = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height1 = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                # Warmup
                if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=opt.augment)[0]
                
                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]
                t2 = time_synchronized()
                # print('ss', len(pred))

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                t3 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        # for c in det[:, -1].unique():
                        #     n = (det[:, -1] == c).sum()  # detections per class
                        #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        dets = []
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            dets.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), int(conf)])
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # det_test = []
                        # for *xyxy, conf, cls in reversed(det):
                        #     det_test.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), int(conf)])
                        #     print(det_test)                            # print(ID)
                            # mm = np.asarray(move[ID])
                            # print(mm)
                            # df = pd.DataFrame(det_test)
                            # df.insert(0, column= None, value = int(cls))
                            # print(df)
                            
                        

                            # if save_img or view_img:  # Add bbox to image
                            #     label = f'{names[int(cls)]} {conf:.2f}'
                                # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
        #----------------------------------------------------------------------------------------------
                        dets = np.asarray(dets)
                        # print(len(dets))
                        if len(dets) > 2:
                            NNI.append(cal_NNI(dets[:, :2], width1, height1))
                            # avg_NNI.append(NNI)
                        try:
                            tracks = tracker.update(dets)
                            # print('destsss', tracks)
                        except:
                            continue
                        boxes = []
                        indexIDs = []
                        c = []
                        previous = memory.copy()
                        memory = {}

                        end = time.time()

                        for track in tracks:
                            # print(track)
                            boxes.append([track[0], track[1], track[2], track[3]])
                            indexIDs.append(int(track[4]))
                            memory[indexIDs[-1]] = boxes[-1]

                            ID = str(int(track[4]))
                            if ID in tracked:
                                tracked[ID].append(track[:4])
                            else:
                                tracked[ID] = [track[:4]]

                            if ID in move:
                                move[ID].append(track[:4])
                            else:
                                move[ID] = [track[:4]]
                            df = pd.DataFrame(move[ID])
                            print(df)
                            # print(ID)
                            # mm = np.asarray(move[ID])
                            # print(mm)
                        # Draw tracking boxes
                        if len(boxes) > 0:
                            i = int(0)
                            for box in boxes:
                                # extract the bounding box coordinates
                                (x, y) = (int(box[0]), int(box[1]))
                                (w, h) = (int(box[2]), int(box[3]))

                                # draw a bounding box rectangle and label on the image
                                # color = [int(c) for c in COLORS[classIDs[i]]]
                                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                                cv2.rectangle(im0s, (x, y), (w, h), color, 2)

                                if indexIDs[i] in previous:
                                    previous_box = previous[indexIDs[i]]
                                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                                    # cv2.line(im0s, p0, p1, color, 3)

                                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                                text = "{}".format(indexIDs[i])
                                cv2.putText(im0s, text, (x, y - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                i += 1
                            cv2.imshow('s', im0s)
                            cv2.waitKey(1)
        #------------------------------------------------------------------------------------------------
                    # Print time (inference + NMS)
                    # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                    # Stream results
                    # if view_img:
                    #     cv2.imshow(str(p), im0)
                    #     cv2.waitKey(1)  # 1 millisecond

                    # # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                            print(f" The image with the result is saved in: {save_path}")
                        else:  # 'video' or 'stream'
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 5, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer.write(im0)
                frameIndex += 1
        # ---------------------------------------------s-------------------------s--------------------
                if frameIndex % 1 == 0:
                    print(frameIndex)
                    movement = cal_movement_hist(move, width1, height1)
                    # movement = [x for x in movement if x < 300 and x != 0]
                    if len(movement) != 0:
                        avg_mov = sum(movement)/len(movement)
                        std_mov = np.std(movement)
                    else:
                        avg_mov = 0
                        std_mov = 0
                # print(movement)
                    nni = sum(NNI)/frameIndex
                    std_nni = np.std(NNI)
                    move = {}
                    avg_movements.append(avg_mov)
                    avg_NNI.append(nni)
                    std_movements.append(std_mov)
                    std_NNI.append(std_nni)
                    NNI = []                  
                with open("/workspace/yolov7/videos/"+video_date+'/'+'aggressive' + "/" + os.path.basename(video).split('.')[0] + ".csv", 'w') as f:
                    csv_writer = csv.writer(f)
                # with open("/workspace/yolov7/videos/" + video_date + '/' + 'aggressive' + "/" + os.path.basename(video).split('.')[0] + ".txt", 'w') as f:
                    for i, x in enumerate(avg_movements):
                        print(x)
                        csv_writer.writerow([str(x)])
            vid_cap.release()
            # ------------------------------------------------------------------------------------------
            print(f'Done. ({time.time() - t0:.3f}s)')
            end = time.time()
            print(end -start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    # print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

    