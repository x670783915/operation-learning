import torch

def diounms(bbox, scores, overlap=0.5, top_k=200, beta1=1.0):
    keep = scores.new(scores.size(0)).zero_().long()
    if bbox.numel() == 0:
        return keep

    x1 = bbox[:, 0]
    x2 = bbox[:, 2]
    y1 = bbox[:, 1]
    y2 = bbox[:, 3]

    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = bbox.new()
    xx2 = bbox.new()
    yy1 = bbox.new()
    yy2 = bbox.new()

    w = bbox.new()
    h = bbox.new()

    cnt = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[cnt] = i
        cnt += 1

        if idx.size(0) = 1:
            break
        
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(y2, 0, idx, out=yy2)

        inx1 = torch.clamp(xx1, min=x1[i])
        iny1 = torch.clamp(yy1, min=y1[i])
        inx2 = torch.clamp(xx2, max=x2[i])
        iny2 = torch.clamp(yy2, max=y2[i])

        center_x1 = (x1[i] + x2[i]) / 2
        center_x2 = (xx1 + xx2) / 2
        center_y1 = (y1[i] + y2[i]) / 2
        center_y2 = (yy1 + yy2) / 2

        d = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

        cx1 = torch.clamp(xx1, max=x1[i])
        cx2 = torch.clamp(xx2, min=x2[i])
        cy1 = torch.clamp(yy1, max=y1[i])
        cy2 = torch.clamp(yy2, min=y2[i])

        c = (cy1 - cy2) ** 2 + (cx1 - cx2) ** 2

        u = d / c

        w.resize_as_(xx2)
        h.resize_as_(yy2)

        w = inx2 - inx1
        h = iny2 - iny1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_ares = torch.index_select(area, 0, idx)
        union = (rem_ares - inter) + area[i]

        iou = inter / union - u ** beta1

        idx = idx[iou.le(overlap)]
    return keep, cnt


