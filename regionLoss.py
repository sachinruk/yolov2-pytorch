import torch
import itertools
import numpy as np
from bbox import *

def Yolov2Loss(output, labels, n_truths, meta):
    B = meta['anchors']
    C = meta['classes']
    batch_size = meta['batch_size']
    threshold = meta['threshold']
    anchor_bias = meta['anchor_bias']
    scale_no_obj = meta['scale_no_obj']
    scale_coords = meta['scale_coords']
    scale_class = meta['scale_class']
    scale_obj = meta['scale_obj']
    
    H = output.size(2)
    W = output.size(3)
    
    wh = torch.from_numpy(np.reshape([W, H], [1, 1, 1, 1, 2])).float()
    anchor_bias_var = torch.from_numpy(np.reshape(anchor_bias, [1, 1, 1, B, 2])).float()
    
    w_list = np.array(list(range(W)), np.float32)
    wh_ids = torch.from_numpy(np.array(list(map(lambda x: np.array(list(itertools.product(w_list, [x]))), range(H)))).reshape(1, H, W, 1, 2)).float() 
    
    zero_pad = torch.zeros(2).contiguous().view(1, 2).float()
    pad_var = torch.zeros(2*B).contiguous().view(B, 2).float()
    
    loss = torch.Tensor([0]).float()
    class_zeros = torch.zeros(C).float()
    mask_loss = torch.zeros(H*W*B*5).contiguous().view(H, W, B, 5).float()
    zero_coords_loss = torch.zeros(H*W*B*4).contiguous().view(H, W, B, 4).float()
    zero_coords_obj_loss = torch.zeros(H*W*B*5).contiguous().view(H, W, B, 5).float()
    
    if torch.cuda.is_available():
        wh = wh.cuda()
        wh_ids = wh_ids.cuda()
        pad_var = pad_var.cuda()
        zero_pad = zero_pad.cuda()
        anchor_bias_var = anchor_bias_var.cuda()
        
        loss = loss.cuda()
        mask_loss = mask_loss.cuda()
        class_zeros = class_zeros.cuda()
        zero_coords_loss = zero_coords_loss.cuda()
        zero_coords_obj_loss = zero_coords_obj_loss.cuda()
                           

    anchor_bias_var = anchor_bias_var / wh
    anchor_padded = torch.cat([pad_var, anchor_bias_var.contiguous().view(B, 2)], 1)

    predicted = output.permute(0, 2, 3, 1)
    predicted = predicted.contiguous().view(-1, H, W, B, (4 + 1 + C))
    
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=4)
    
    adjusted_xy = sigmoid(predicted[:, :, :, :, :2])
    adjusted_obj = sigmoid(predicted[:, :, :, :, 4:5])
    adjusted_classes = softmax(predicted[:, :, :, :, 5:])
    
    adjusted_coords = (adjusted_xy + wh_ids) / wh

    adjusted_wh = torch.exp(predicted[:, :, :, :, 2:4]) * anchor_bias_var
    
    for batch in range(batch_size):
        
        n_true = n_truths[batch]
        if n_true == 0:
            continue

        pred_outputs = torch.cat([adjusted_coords[batch], adjusted_wh[batch]], 3)
        true_labels = labels[batch, :n_true, 1:]
        
        bboxes_iou = bbox_overlap_iou(pred_outputs, true_labels, False)
        
        # objectness loss (if iou < threshold)
        boxes_max_iou = torch.max(bboxes_iou, -1)[0] #is this necessary?
        all_obj_mask = boxes_max_iou.le(threshold)
        all_obj_loss = all_obj_mask.unsqueeze(-1).float() *(scale_no_obj * (-1 * adjusted_obj[batch]))
        
        # each anchor box will learn its bias (if batch < 12800)
        all_coords_loss = zero_coords_loss.clone()
        if meta['iteration'] < 12800:
            all_coords_loss = scale_coords * torch.cat([(0.5 - adjusted_xy[batch]), (0 - predicted[batch, :, :, :, 2:4])], -1)
        
        coord_obj_loss = torch.cat([all_coords_loss, all_obj_loss], -1)
        
        batch_mask = mask_loss.clone()
        truth_coord_obj_loss = zero_coords_obj_loss.clone()
        # for every true label and anchor bias
#         from IPythonx.core.debugger import Tracer; Tracer()()
        for truth_iter in torch.arange(n_true):
            truth_iter = int(truth_iter)
            truth_box = labels[batch, truth_iter]
            anchor_select = bbox_overlap_iou(torch.cat([zero_pad.t(), truth_box[3:].unsqueeze(-1)], 0).t(), anchor_padded, True)
            
            # find the responsible anchor box
            anchor_id = torch.max(anchor_select, 1)[1]
            
            truth_i = (truth_box[1] * W)
            w_i = truth_i.int() 
            truth_x = truth_i - w_i.float()
            truth_j = (truth_box[2] * H)
            h_j = truth_j.int()
            truth_y = truth_j - h_j.float()
            truth_wh = (truth_box[3:] / anchor_bias_var.contiguous().view(B, 2).index_select(0, anchor_id)).log()

            truth_coords = torch.cat([truth_x.view(1,1), truth_y.view(1,1), truth_wh], 1)
            
            predicted_output = predicted[batch, h_j, w_i, anchor_id].squeeze()
            # coords loss
            pred_xy = adjusted_xy[batch, h_j, w_i, anchor_id].squeeze()
            pred_wh = predicted_output[2:4]
            pred_coords = torch.cat([pred_xy, pred_wh], 0)
            coords_loss = scale_coords * (truth_coords - pred_coords.unsqueeze(0))
    
            # objectness loss
        
            # given the responsible box - find iou
            iou = bboxes_iou[h_j, w_i, anchor_id, truth_iter].squeeze()
            obj_loss = scale_obj * (iou - sigmoid(predicted_output[4]))
            truth_co_obj = torch.cat([coords_loss, obj_loss.view(1, 1)], 1)

            # class prob loss
            class_vec = class_zeros.index_fill(0, truth_box[0].long(), 1)
            class_loss = scale_class * (class_vec - torch.nn.Softmax(dim=0)(predicted_output[5:]))
            
            mask_ones = torch.ones(5).float().cuda()
            
            batch_mask[h_j, w_i, anchor_id] = mask_ones
            truth_coord_obj_loss[h_j, w_i, anchor_id] = truth_co_obj
            
            loss += class_loss.pow(2).sum()
        batch_coord_obj_loss = batch_mask * truth_coord_obj_loss + (1 - batch_mask) * coord_obj_loss
        
        loss += batch_coord_obj_loss.pow(2).sum()
        
    return loss