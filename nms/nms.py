import numpy as np

def nms(dets, thresh=0.5):
    x1 = [:, 0]
    y1 = [:, 1]
    x2 = [:, 2]
    y2 = [:, 3]
    scores = [:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    keep = []
    _, idx = scores.argsort()[::-1].astype(np.int32)

    while idx.size() > 0:
        i = idx[0]
        idx = idx[1:]
        keep.append(i)

        inx1 = np.maximum(x1[i], x1[idx])
        inx2 = np.minimum(x2[i], x2[idx])
        iny1 = np.maximum(y1[i], y1[idx])
        iny2 = np.minimum(y2[i], y2[idx])

        w, h = (inx2 - inx1 + 1), (iny2 - iny1 + 1)
        w, h = np.maximum(0, w), np.maximum(0, h)
        inter = w * h
        iou = inter / (areas[i] + areas[idx] - inter)

        idx = idx[np.where(iou <= thresh)[0]]
    return keep

def diou_nms(dets, thresh=0.5, beta1=1):
    x1 = [:, 0]
    y1 = [:, 1]
    x2 = [:, 2]
    y2 = [:, 3]
    scores = [:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    keep = []
    _, idx = scores.argsort()[::-1].astype(np.int32)

    while idx.size() > 0:
        i = idx[0]
        idx = idx[1:]
        keep.append(i)

        inx1 = np.maximum(x1[i], x1[idx])
        inx2 = np.minimum(x2[i], x2[idx])
        iny1 = np.maximum(y1[i], y1[idx])
        iny2 = np.minimum(y2[i], y2[idx])

        w, h = (inx2 - inx1 + 1), (iny2 - iny1 + 1)
        w, h = np.maximum(0, w), np.maximum(0, h)
        inter = w * h
        iou = inter / (areas[i] + areas[idx] - inter)

        outx1 = np.minimum(x1[i], x1[idx])
        outx2 = np.maximum(x2[i], x2[idx])
        outy1 = np.minimum(y1[i], y1[idx])
        outy2 = np.maximum(y2[i], y2[idx])

        center_i_x = (x1[i] + x2[i]) / 2
        center_i_y = (y1[i] + y2[i]) / 2
        center_x = (outx1 + outx2) / 2
        center_y = (outy1 + outy2) / 2
        d = (center_y - center_i_y) ** 2 + (center_i_x - center_x) ** 2
        c = (outy2 - outy1) ** 2 + (outx1 - outx2) ** 2

        diou = iou - (d / c) ** beta1

        idx = idx[np.where(diou <= thresh)[0]]
    return keep