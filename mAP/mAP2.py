import numpy as np

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i], mpre[i-1])

    i = np.where(mrec[1:] != mrec[:-1])[0] # 错开一位看计算存在变化的位置

    ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1]) # 计算面积
    return ap


def voc_ap(rec, rec, pre, use_07_metric=True):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(pre[rec >= t])
            ap += p / 11.
    else:
        mpre = np.concatenate(([0.0], pre, [0.0]))
        mrec = np.concatenate(([0.0], rec, [1.0]))

        #保证precision从前往后非减
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i], mpre[i-1])
        
        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap


