# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# written by allenjbqin
# 2019.05.15
import os
import numpy as np
import ctypes
from ctypes import *

so_file_path = os.path.join( os.path.abspath(os.path.dirname(__file__)), 'librbox.so')
so = ctypes.cdll.LoadLibrary
librbox = so(so_file_path)

overlap = librbox.Overlap
overlap.argtypes = (POINTER(c_double), POINTER(c_double))
overlap.restype = c_double


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def angle_soft_nms(all_dets, sigma=0.5, Nt=0.1, threshold=0.001, method=0):
    """Pure Python NMS baseline."""
    # dets = np.concatenate((all_dets[:, 0:4], all_dets[:, -1:]), axis=1)
    # scores = all_dets[:, 4]
    # cx,cy,w,h,angle,score
    # all_dets[:,4] = (all_dets[:,4]-0.5)*180.0
    all_dets[:,4] = all_dets[:,4] / np.pi *180.0
    boxes = all_dets
    # scores = all_dets[:, -1]
    N = all_dets.shape[0]
    if N >0 and len(boxes[0] == 7):
        is_scale = True
    else:
        is_scale = False

    # # ## a simple example
    # a = np.array([10,10,10,20,0])
    # b = np.array([10,10,10,20,0.75])
    # cd_a =  (c_double * 5)()
    # cd_b =  (c_double * 5)()
    # cd_a[0] = c_double(a[0])
    # cd_a[1] = c_double(a[1])
    # cd_a[2] = c_double(a[2])
    # cd_a[3] = c_double(a[3])
    # cd_a[4] = c_double(a[4])
    # cd_b[0] = c_double(b[0])
    # cd_b[1] = c_double(b[1])
    # cd_b[2] = c_double(b[2])
    # cd_b[3] = c_double(b[3])
    # cd_b[4] = c_double(b[4])
    # ov = overlap(cd_a, cd_b)
    for i in range(N):
        maxscore = boxes[i, 5]
        maxpos = i
        # 将第i个bbox存在temp
        tcx = boxes[i, 0]
        tcy = boxes[i, 1]
        tw = boxes[i, 2]
        th = boxes[i, 3]
        tangle = boxes[i, 4]
        ts= boxes[i, 5]
        if is_scale:
            scale= boxes[i, 6]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 5]:
                maxscore = boxes[pos, 5]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]
        boxes[i, 5] = boxes[maxpos, 5]
        if is_scale:
            boxes[i, 6] = boxes[maxpos, 6]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tcx
        boxes[maxpos, 1] = tcy
        boxes[maxpos, 2] = tw
        boxes[maxpos, 3] = th
        boxes[maxpos, 4] = tangle
        boxes[maxpos, 5] = ts
        if is_scale:
            boxes[maxpos, 6] = scale

        # 此时第i个位最大score的，重新将第i个bbox存在temp
        # tcx = boxes[i, 0]
        # tcy = boxes[i, 1]
        # tw = boxes[i, 2]
        # th = boxes[i, 3]
        # tangle = boxes[i, 4]
        # ts= boxes[i, 5]

        box1 = (c_double * 5)()
        box2 = (c_double * 5)()
        box1[0] = c_double(boxes[i, 0])
        box1[1] = c_double(boxes[i, 1])
        box1[2] = c_double(boxes[i, 2])
        box1[3] = c_double(boxes[i, 3])
        box1[4] = c_double(boxes[i, 4])

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            box2[0] = c_double(boxes[pos, 0])
            box2[1] = c_double(boxes[pos, 1])
            box2[2] = c_double(boxes[pos, 2])
            box2[3] = c_double(boxes[pos, 3])
            box2[4] = c_double(boxes[pos, 4])

            ov = overlap(box1, box2)
            if ov > 0:
                if method == 1:  # linear
                    if ov > Nt:
                        weight = 1 - ov
                    else:
                        weight = 1
                elif method == 2:  # gaussian
                    weight = np.exp(-(ov * ov) / sigma)
                else:  # original NMS
                    if ov > Nt:
                        weight = 0
                    else:
                        weight = 1

                boxes[pos, 5] = weight * boxes[pos, 5]

                # if box score falls below threshold, discard the box by swapping with last box
                # update N
                if boxes[pos, 5] < threshold:
                    boxes[pos, 0] = boxes[N - 1, 0]
                    boxes[pos, 1] = boxes[N - 1, 1]
                    boxes[pos, 2] = boxes[N - 1, 2]
                    boxes[pos, 3] = boxes[N - 1, 3]
                    boxes[pos, 4] = boxes[N - 1, 4]
                    boxes[pos, 5] = boxes[N - 1, 5]
                    if is_scale:
                        boxes[pos, 6] = boxes[N - 1, 6]
                    N = N - 1
                    pos = pos - 1
            pos = pos + 1
    keep = [i for i in range(N)]
    # boxes[:, 4] = (boxes[:, 4] / 180.0) + 0.5
    boxes[:, 4] = boxes[:, 4] / 180.0 * np.pi
    return boxes

def angle_soft_nms_new(all_dets, sigma=0.5, Nt=0.5, threshold=0.03, method=0, all_cls=False, cls_decay=1.5):
    """Pure Python Soft-NMS baseline.
        author: Xingjia Pan
        date: 2019/11/4
        all_dets: cx,cy,w,h,angle,score for one row
    """
    all_dets[:,4] = all_dets[:,4] / np.pi *180.0
    N = all_dets.shape[0]
    for i in range(N):
        order = np.argsort(-all_dets[:, 5])
        all_dets = all_dets[order, :]
        ##  calc distance of center point
        if i == N-1:
            continue
        dist_score = np.linalg.norm(all_dets[i,:2]-all_dets[i+1:, :2],axis=1)
        min_side = np.min(all_dets[i,2:4])+1e-8
        div_factor = 1./10 if min_side>96 else 1./7
        dist_score = dist_score/(div_factor * min_side)
        dist_score = np.clip(dist_score, 0.0, 1.0)
        dist_score = dist_score**2
        box1 = (c_double * 5)()
        box1[0] = c_double(all_dets[i, 0])
        box1[1] = c_double(all_dets[i, 1])
        box1[2] = c_double(all_dets[i, 2])
        box1[3] = c_double(all_dets[i, 3])
        box1[4] = c_double(all_dets[i, 4])
        j = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while j < N:
            box2 = (c_double * 5)()
            box2[0] = c_double(all_dets[j, 0])
            box2[1] = c_double(all_dets[j, 1])
            box2[2] = c_double(all_dets[j, 2])
            box2[3] = c_double(all_dets[j, 3])
            box2[4] = c_double(all_dets[j, 4])
            ov = overlap(box1, box2)
            weight = 1.0
            if ov > 0:
                if method == 1:  # linear
                    if ov > Nt:
                        weight = 1 - ov
                    else:
                        weight = 1
                elif method == 2:  # gaussian
                    weight = np.exp(-(ov * ov) / sigma)
                else:  # original NMS
                    if ov > Nt:
                        weight = 0
                    else:
                        weight = 1
            if all_cls:
                if all_dets[i,6] != all_dets[j, 6]:
                    dist_score[j-i-1] *= cls_decay
                    dist_score[j-i-1] = np.minimum(dist_score[j-i-1], 1.0)
            weight *= dist_score[j-i-1]
            all_dets[j, 5] = weight * all_dets[j, 5]
            j = j + 1
    keep = all_dets[:,5] > threshold
    all_dets[:, 4] = all_dets[:, 4] / 180.0 * np.pi
    return all_dets[keep,:]


def py_yt_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]

    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    angle = 180.0 * (dets[:, 5] - 0.5)

    # sort by confidence
    order = scores.argsort()[::-1]

    # list of keep box
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        cur_box = [x1[i], y1[i], x2[i], y2[i], angle[i]]
        other_boxes = []
        for each_box in range(1, len(order)):
            each_other_box = [x1[order[each_box]], y1[order[each_box]], x2[order[each_box]], y2[order[each_box]],angle[order[each_box]]]
            other_boxes.append(each_other_box)
        iou_result_list = []
        box1 = (c_double * 5)()
        box2 = (c_double * 5)()
        box1[0] = c_double(cur_box[0])
        box1[1] = c_double(cur_box[1])
        box1[2] = c_double(cur_box[2])
        box1[3] = c_double(cur_box[3])
        box1[4] = c_double(cur_box[4])
        # call for cpp nms function
        for each_gt_box in other_boxes:
            box2[0] = c_double(each_gt_box[0])
            box2[1] = c_double(each_gt_box[1])
            box2[2] = c_double(each_gt_box[2])
            box2[3] = c_double(each_gt_box[3])
            box2[4] = c_double(each_gt_box[4])

            # get return iou result
            each_iou = overlap(box1, box2)
            iou_result_list.append(each_iou)

        ovr = iou_result_list
        ovr = np.array(ovr)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


if __name__ == '__main__':
    a = np.array([(0, 0, 30 / 1000, 10 / 1000, .9, 0),
                  (0, 0, 30 / 1000, 10 / 1000, .98, 0.25)])  # , (-5, -5, 5, 5, .98, 45), (-5, -5, 6, 6, .99, 30)])
    # print(py_cpu_nms(a, 0.45))
    # print(py_poly_nms(a, 0.45))
    # print(Polygon(a).area)
