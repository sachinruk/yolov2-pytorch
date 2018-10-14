import itertools
import torch
import numpy as np
import cv2

def get_nms_boxes(output, obj_thresh, iou_thresh, meta):
#     import pdb; pdb.set_trace()
    N, C, H, W = output.size()
    N, C, H, W = int(N), int(C), int(H), int(W)
    B = meta['anchors']
    anchor_bias = meta['anchor_bias']
    n_classes = meta['classes']
    
    HWB = H*W*B
    # -1 => unprocesse, 0 => suppressed, 1 => retained
    box_tags = -1 * torch.ones(HWB).float()
    
    wh = torch.from_numpy(np.reshape([W, H], [1, 1, 1, 1, 2])).float()
    anchor_bias_var = torch.from_numpy(np.reshape(anchor_bias, [1, 1, 1, B, 2])).float()
    
    w_list = np.array(list(range(W)), np.float32)
    wh_ids = torch.from_numpy(np.array(list(map(lambda x: np.array(list(itertools.product(w_list, [x]))), range(H)))).reshape(1, H, W, 1, 2)).float() 
    
    if torch.cuda.is_available():
        wh = wh.cuda()
        wh_ids = wh_ids.cuda()
        box_tags = box_tags.cuda()
        anchor_bias_var = anchor_bias_var.cuda()                           

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

    scoresN = (adjusted_obj * adjusted_classes).contiguous().view(N,HWB, -1)
    class_probsN = adjusted_classes.contiguous().view(N, HWB, -1)
    class_idsN = torch.max(class_probsN, -1)[1]
    pred_outputsN = torch.cat([adjusted_coords, adjusted_wh], -1)
    pred_bboxesN = pred_outputsN.contiguous().view(N, HWB, -1)
    condfidencesN = adjusted_obj.contiguous().view(N, HWB)

    idx = scoresN>obj_thresh
    pred_boxes = [pred_bboxesN[idx[...,i]] for i in range(n_classes)]
    scores = [scoresN[...,i][idx[...,i]] for i in range(n_classes)]
    for n in range(N):
        scores = scoresN[n]
        class_probs = class_probsN[n]
        class_ids = class_idsN[n]
            
        # pred_outputs = pred_outputsN[n]
        pred_bboxes = pred_bboxesN[n]
        ious = bbox_overlap_iou(pred_bboxes, pred_bboxes, True) < iou_thresh
        
        confidences = condfidencesN[n]
        # get all boxes with tag -1
        final_boxes = torch.FloatTensor()
        if torch.cuda.is_available():
            final_boxes = final_boxes.cuda()

        for class_id in range(n_classes):
            bboxes_state = -((class_ids==class_id) * (scores[:, class_id] > obj_thresh)).long().float()
        
            while (torch.sum(bboxes_state==-1) > 0).data[0]:
                max_conf, index = torch.max(scores[:, class_id] * (bboxes_state==-1).float(), 0)
                # different bboxes state to above
                bboxes_state = (ious[index][0].float() * bboxes_state).long().float()
                bboxes_state[index] = 1

                index_vals = torch.cat([pred_bboxes[index], confidences[index].view(1, 1), class_probs[index]], 1)
                if len(final_boxes) == 0:
                    final_boxes = index_vals
                else:
                    final_boxes = torch.cat([final_boxes, index_vals], 0)
        
        batch_boxes[n] = final_boxes
        
    return batch_boxes