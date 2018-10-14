import torch

def bbox_overlap_iou(bboxes1, bboxes2, is_anchor):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.
        p1 *-----        
           |     |
           |_____* p2
    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """
#     import pdb; pdb.set_trace()
    x1, y1, w1, h1 = bboxes1.chunk(4, dim=-1)
    x2, y2, w2, h2 = bboxes2.chunk(4, dim=-1)
    
    x11 = x1 - 0.5*w1
    y11 = y1 - 0.5*h1
    x12 = x1 + 0.5*w1
    y12 = y1 + 0.5*h1
    x21 = x2 - 0.5*w2
    y21 = y2 - 0.5*h2
    x22 = x2 + 0.5*w2
    y22 = y2 + 0.5*h2
    
#     x11 = torch.clamp(x11, min=0, max=1)
#     y11 = torch.clamp(y11, min=0, max=1)
#     x12 = torch.clamp(x12, min=0, max=1)
#     y12 = torch.clamp(y12, min=0, max=1)
#     x21 = torch.clamp(x21, min=0, max=1)
#     y21 = torch.clamp(y21, min=0, max=1)
#     x22 = torch.clamp(x22, min=0, max=1)
#     y22 = torch.clamp(y22, min=0, max=1)
    

    xI1 = torch.max(x11, x21.transpose(1, 0))
    yI1 = torch.max(y11, y21.transpose(1, 0))
    
    xI2 = torch.min(x12, x22.transpose(1, 0))
    yI2 = torch.min(y12, y22.transpose(1, 0))

    inner_box_w = torch.clamp((xI2 - xI1), min=0)
    inner_box_h = torch.clamp((yI2 - yI1), min=0)
    
    inter_area = inner_box_w * inner_box_h
    bboxes1_area = (x12 - x11) * (y12 - y11)
    bboxes2_area = (x22 - x21) * (y22 - y21)

    union = (bboxes1_area + bboxes2_area.transpose(1, 0)) - inter_area
    return torch.clamp(inter_area / union, min=0)
