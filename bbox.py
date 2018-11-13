import torch
import numpy as np
from collections import defaultdict

def bbox_overlap_iou(bboxes1, bboxes2):
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


def get_nms_boxes(output, obj_thresh, iou_thresh, meta):
#     import pdb; pdb.set_trace()
    N, C, H, W = output.size()
    B = meta['anchors']
    anchor_bias = meta['anchor_bias']
    n_classes = meta['classes']
    
    HWB = H*W*B
    
    wh = torch.from_numpy(np.reshape([W, H], [1, 1, 1, 1, 2])).float().cuda()
    wh_ids = torch.from_numpy(np.stack(np.meshgrid(np.arange(W), np.arange(H)), 2).reshape((1,W,H,1,2))).float().cuda()
    anchor_bias_var = torch.from_numpy(np.reshape(anchor_bias, [1, 1, 1, B, 2])).float().cuda()                         
    anchor_bias_var = anchor_bias_var / wh

    predicted = output.permute(0, 2, 3, 1)
    predicted = predicted.contiguous().view(N, H, W, B, -1)
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=4)
    
    adjusted_xy = sigmoid(predicted[:, :, :, :, :2])
    adjusted_obj = sigmoid(predicted[:, :, :, :, 4:5])
    adjusted_classes = softmax(predicted[:, :, :, :, 5:])
    
    adjusted_coords = (adjusted_xy + wh_ids) / wh
    adjusted_wh = torch.exp(predicted[:, :, :, :, 2:4]) * anchor_bias_var

    batch_boxes = defaultdict()
    batch_classes = defaultdict()

    scoresN = (adjusted_obj * adjusted_classes).contiguous().view(N,HWB, -1)
    # class_probsN = adjusted_classes.contiguous().view(N, HWB, -1)
    # class_idsN = torch.max(class_probsN, -1)[1]
    pred_outputsN = torch.cat([adjusted_coords, adjusted_wh], -1)
    pred_bboxesN = pred_outputsN.contiguous().view(N, HWB, -1)
    # condfidencesN = adjusted_obj.contiguous().view(N, HWB)

    idx = scoresN>obj_thresh
    # pred_boxes = [pred_bboxesN[idx[...,i]] for i in range(n_classes)]
    # scores = [scoresN[...,i][idx[...,i]] for i in range(n_classes)]
    pred_boxes = [[pred_bboxesN[num][idx[num,:,i]] for i in range(n_classes)] for num in range(N)]
    scores = [[scoresN[num,:,i][idx[num,:,i]] for i in range(n_classes)] for num in range(N)]

    
    for n in range(N):
        # scores = scoresN[n]
        # class_probs = class_probsN[n]
        # class_ids = class_idsN[n]
        # confidences = condfidencesN[n]

        final_boxes = torch.FloatTensor().cuda()
        class_assignment = []
        prev_boxes = 0
        for class_id in range(n_classes):
            
            while len(pred_boxes[n][class_id]) > 0:
                # get bounding box with higest score
                pred_bboxes = pred_boxes[n][class_id]
                idx_max = torch.argmax(scores[n][class_id])
                ious = bbox_overlap_iou(pred_bboxes[idx_max], pred_bboxes, True) < iou_thresh
                
                # append this box to list of bounding boxes to save
                index_vals = torch.cat([pred_bboxes[idx_max], scores[n][class_id][idx_max].view(1)]).unsqueeze(0)
                if len(final_boxes) == 0:
                    final_boxes = index_vals
                else:
                    final_boxes = torch.cat([final_boxes, index_vals], 0)

                # get rid of all boxes that overlap with this box too much
                pred_boxes[n][class_id] = pred_boxes[n][class_id][ious.squeeze()]
                scores[n][class_id] = scores[n][class_id][ious.squeeze()]
            class_assignment += [class_id]*(len(final_boxes) - prev_boxes)
            prev_boxes = len(final_boxes)

        batch_boxes[n] = final_boxes
        batch_classes[n] = class_assignment
        
    return batch_boxes, batch_classes


def calc_map(ground_truths, nms_boxes, nms_classes, n_truths, iou_thresh):
    
    N = ground_truths.size(0)
    
    mean_avg_precision = torch.FloatTensor([0]).cuda()

    for batch in range(N):
        category_map = defaultdict(lambda: defaultdict(lambda: torch.FloatTensor().cuda()))
        
        if n_truths[batch] == 0 or len(nms_boxes[batch]) == 0:
            continue

        # break the ground truth into groups of given classes of boxes
        for gt in ground_truths[batch, :n_truths[batch]]:
            gt_class = int(gt[0])
            t1 = category_map[gt_class]['ground_truth']
            t1 = torch.cat([t1, gt[1:].unsqueeze(0)], 0)
            category_map[gt_class]['ground_truth'] = t1

        for box, class_id in zip(nms_boxes[batch], nms_classes[batch]):
            t2 = category_map[class_id]['prediction']
            t2 = torch.cat([t2, box.unsqueeze(0)], 0)
            category_map[class_id]['prediction'] = t2
        cat_ids = category_map.keys()
        
        ap_per_category = [calc_map_(category_map[cat_id], iou_thresh) for cat_id in cat_ids]
        mean_avg_precision += torch.mean(torch.cat(ap_per_category, 0))
    return mean_avg_precision/N


def calc_map_(boxes_dict, iou_threshold=0.5):
#     import pdb; pdb.set_trace()
    if len(boxes_dict['ground_truth'])==0 or len(boxes_dict['prediction'])==0:
        return torch.zeros(1).cuda()

    gt = boxes_dict['ground_truth']
    pr = boxes_dict['prediction']

    import pdb; pdb.set_trace()
    gt_matched = -torch.ones(gt.size(0)).cuda()
    pr_matched = -torch.ones(pr.size(0)).cuda()
            
    for i in range(len(pr)):
        b = pr[i]
        ious = bbox_overlap_iou(b[:4].view(1, 4), gt)
        matched_scores = (gt_matched == -1).float() * (ious[0]>iou_threshold).float() * ious[0]
        if torch.sum(matched_scores) > 0:
            gt_idx = torch.max(matched_scores, 0)[1]
            gt_matched[gt_idx] = i
            pr_matched[i] = gt_idx
        
    tp = (pr_matched != -1).float()
    fp = (pr_matched == -1).float()
    tp_cumsum = torch.cumsum(tp, 0)
    fp_cumsum = torch.cumsum(fp, 0)
    n_corrects = tp_cumsum * tp
    total = tp_cumsum + fp_cumsum
    precision = n_corrects / total
    for i in range(precision.size(0)):
        precision[i] = torch.max(precision[i:])

    average_precision = torch.sum(precision) / len(gt)
    return average_precision.unsqueeze(0)