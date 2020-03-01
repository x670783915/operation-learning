import numpy as np

def py_cpu_nms(dets, thres=0.5):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []

    index = (scores.argsort()[::-1]).astype(np.int32)

    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thres)[0]

        index = index[idx+1] # index staty with 1

    return keep

import matplotlib.pyplot as plt

def plot_bbox(dets, c='k'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)

if __name__ == '__main__':

    # fake data
    boxes = np.array([
        [100,100,210,210,0.72],
        [250,250,420,420,0.8],
        [220,220,320,330,0.92],
        [100,100,210,210,0.72],
        [230,240,325,330,0.81],
        [220,230,315,340,0.9]
    ])

    plot_bbox(boxes, 'k')
    plt.show()
    keep = py_cpu_nms(boxes, thres=0.7)
    plot_bbox(boxes[keep], 'r')
    plt.show()