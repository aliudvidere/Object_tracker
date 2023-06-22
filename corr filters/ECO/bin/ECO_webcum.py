import glob
import time
import os
import pandas as pd
import argparse
import numpy as np
import cv2
import sys
sys.path.append('./')

from eco import ECOTracker
from PIL import Image

import argparse

def main(video_dir, video_flag, gt_flag):
    # load videos
    if (not video_flag):
        filenames = sorted(glob.glob(os.path.join(video_dir, "img/*.jpg")),
            key=lambda x: int(os.path.basename(x).split('.')[0]))
        # frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
        frames = [np.array(Image.open(filename)) for filename in filenames]
    else:
        video = cv2.VideoCapture(0)
        # load video
        if not video.isOpened():
            print('[ERROR] video file not loaded')
            sys.exit()
        # capture first frame
        ok, frame = video.read()
        if not ok:
            print('[ERROR] no frame captured')
            sys.exit()
        print('[INFO] video loaded and frame capture started')
    height, width = frame.shape[:2]
    if len(frame.shape) == 3:
        is_color = True
    else:
        is_color = False
        frames = [frame[:, :, np.newaxis] for frame in frames]
    if gt_flag:
        gt_bboxes = pd.read_csv(os.path.join(video_dir, "groundtruth_rect.txt"), sep='\t|,| ',
                header=None, names=['xmin', 'ymin', 'width', 'height'],
                engine='python')

    title = video_dir.split('/')[-1]
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # img_writer = cv2.VideoWriter(os.path.join('./videos', title+'.avi'),
    #         fourcc, 25, (width, height))
    # starting tracking
    tracker = ECOTracker(is_color)
    vis = True
    # video = cv2.VideoCapture(0)  # Use 0 for the default camera

    # Allow the camera to warm up by waiting for a moment
    time.sleep(1)

    # Read a single frame from the video stream
    ret, frame = video.read()

    # Check if the frame is read correctly
    if not ret:
        print("Failed to read frame")
        exit()

    # Display the captured frame
    cv2.imshow("Captured Frame", frame)

    # Wait for the "Esc" key press to exit
    # while cv2.waitKey(0) != 27:  # 27 corresponds to the "Esc" key
    #     pass

    # Release the VideoCapture object
    video.release()

    # Close windows
    # cv2.destroyAllWindows()

    # cv2.waitKey(1)
    bbox = cv2.selectROI(frame)
    print('[INFO] select ROI and press ENTER or SPACE')
    print('[INFO] cancel selection by pressing C')
    print(bbox)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    tracker.init(frame, bbox)
    bbox = (bbox[0]-1, bbox[1]-1,
            bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
    idx = 0
    video = cv2.VideoCapture(0)
    # load video
    if not video.isOpened():
        print('[ERROR] video file not loaded')
        sys.exit()
    while True:
        ok, frame = video.read()
        if not ok:
            print('[INFO] end of video file reached')
            break
        if idx == 0:
            # bbox = cv2.selectROI(frame)
            # print('[INFO] select ROI and press ENTER or SPACE')
            # print('[INFO] cancel selection by pressing C')
            # print(bbox)
            # cv2.destroyAllWindows()
            # cv2.waitKey(1)
            # tracker.init(frame, bbox)
            # bbox = (bbox[0]-1, bbox[1]-1,
            #         bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
            pass
        bbox = tracker.update(frame, True, vis)
        # bbox xmin ymin xmax ymax
        frame = frame.squeeze()
        if len(frame.shape) == 3:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pass
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 255),
                              1)
        if gt_flag:
            gt_bbox = gt_bboxes.iloc[idx].values
            gt_bbox = (gt_bbox[0], gt_bbox[1],
                    gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3])
            frame = frame.squeeze()
            frame = cv2.rectangle(frame,
                                (int(gt_bbox[0]-1), int(gt_bbox[1]-1)), # 0-index
                                (int(gt_bbox[2]-1), int(gt_bbox[3]-1)),
                                (0, 255, 0),
                                1)
        if vis and idx > 0:
            score = tracker.score
            size = tuple(tracker.crop_size.astype(np.int32))
            score = cv2.resize(score, size)
            score -= score.min()
            score /= score.max()
            score = (score * 255).astype(np.uint8)
            # score = 255 - score
            score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
            pos = tracker._pos
            pos = (int(pos[0]), int(pos[1]))
            xmin = pos[1] - size[1]//2
            xmax = pos[1] + size[1]//2 + size[1] % 2
            ymin = pos[0] - size[0] // 2
            ymax = pos[0] + size[0] // 2 + size[0] % 2
            left = abs(xmin) if xmin < 0 else 0
            xmin = 0 if xmin < 0 else xmin
            right = width - xmax
            xmax = width if right < 0 else xmax
            right = size[1] + right if right < 0 else size[1]
            top = abs(ymin) if ymin < 0 else 0
            ymin = 0 if ymin < 0 else ymin
            down = height - ymax
            ymax = height if down < 0 else ymax
            down = size[0] + down if down < 0 else size[0]
            score = score[top:down, left:right]
            crop_img = frame[ymin:ymax, xmin:xmax]
            # if crop_img.shape != score.shape:
            #     print(left, right, top, down)
            #     print(xmin, ymin, xmax, ymax)
            score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
            frame[ymin:ymax, xmin:xmax] = score_map

        frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        # img_writer.write(frame)
        cv2.imshow(title, frame)
        cv2.waitKey(1)
        idx += 1
        if cv2.waitKey(20) == 27: # Клавиша Esc
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type = str, default='sequences/Crossing/')
    parser.add_argument('--video_flag', type = bool, default=True)
    parser.add_argument('--gt_flag', type = bool, default=False)
    args = parser.parse_args()
    main(args.video_dir, args.video_flag, args.gt_flag)
