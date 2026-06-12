import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

from sort import Sort   # 🔥 NEW IMPORT

# ───────────────────────────────────────────────

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source == '0' or source == '1'
    if source.endswith(('.mp4', '.avi', '.mkv')):
        webcam = False
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)


    
    if half:
        model.half()
    
    from sort import Sort
    

    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()

    # 🔥 SORT TRACKER
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    seen_total = set()
    vid_path = None
    vid_writer = None

    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        for i, det in enumerate(pred):

            if webcam:
                im0 = im0s[i].copy()
            else:
                im0 = im0s

            head_boxes = []

            head_count = 0

            if len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:

                    label = names[int(cls)]

                    if 'head' in label.lower():

                        x1, y1, x2, y2 = map(int, xyxy)

                        head_boxes.append([x1, y1, x2, y2, float(conf)])

                        head_count += 1

                        plot_one_box(xyxy, im0,
                                     label=f"{label} {conf:.2f}",
                                     color=(0, 255, 0),
                                     line_thickness=2)

                    else:
                        plot_one_box(xyxy, im0,
                                     label=f"{label} {conf:.2f}",
                                     color=(255, 0, 0),
                                     line_thickness=2)

            # 🔥 TRACKING STEP (SORT)
            if len(head_boxes) == 0:
                tracked = np.empty((0, 5))
            else:
                tracked = tracker.update(np.array(head_boxes))

            for x1, y1, x2, y2, oid in tracked:

                oid = int(oid)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                seen_total.add(oid)

                cv2.circle(im0, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(im0, f"ID {oid}", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)

            # ── UI Overlay ──
            cv2.rectangle(im0, (10, 10), (500, 110), (0, 0, 0), -1)

            cv2.putText(im0,
                        f"TOTAL UNIQUE: {len(seen_total)}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            cv2.putText(im0,
                        f"IN FRAME: {head_count}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (200, 200, 200), 2)

            print(f"Done ({t2 - t1:.3f}s)")

            if view_img:
                cv2.imshow("Tracking", im0)
                cv2.waitKey(1)   # 🔥 FIXED

            if save_img:
                save_path = str(save_dir / "output.mp4")

                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)

                else:

                    if vid_writer is None or vid_path != save_path:
                        vid_path =  save_path

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        if fps == 0 or fps is None:
                            fps = 25

                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                        
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

                    vid_writer.write(im0)

    if vid_writer is not None: 
         vid_writer.release()             

    print(f"Done. Total time: {time.time() - t0:.3f}s")


# ───────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt')
    parser.add_argument('--source', type=str, default='data/images')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='')
    parser.add_argument('--view-img', action='store_true')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')

    opt = parser.parse_args()
    check_requirements()

    with torch.no_grad():
        detect(save_img=True)