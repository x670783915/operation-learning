import numpy as np
import time
from nms_py2 import py_cpu_nms

np.random.seed( 1 )
num_rois = 6000
minxy = np.random.randint(50, 145, size=(num_rois, 2))
maxxy = np.random.randint(150, 200, size=(num_rois, 2))
score = 0.8 * np.random.random_sample((num_rois, 1)) + 0.2

boxes_new = np.concatenate((minxy, maxxy, score), axis=1).astype(np.float32)

def nms_test_time(boxes_new):
    thres = [0.5, 0.7, 0.8, 0.9]
    T = 50
    for i in range(len(thres)):
        since = time.time()
        for t in range(T):
            keep = py_cpu_nms(boxes_new, thres[i])
        print('thres = {:.1f}, time = {:.4f}'.format(thres[i], (time.time() - since) / T))
    return keep

if __name__ == '__main__':
    nms_test_time(boxes_new)
