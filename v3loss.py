import torch
import torch.nn as nn
import numpy as np
from bbox import bbox_overlap_iou

class Yolov3Loss:
    def set_params(self, shape, meta):
        self.B = meta['anchors']
        self.C = meta['classes']
        self.threshold = meta['threshold']
        self.anchor_bias = meta['anchor_bias']
        self.scale_no_obj = meta['scale_no_obj']
        self.scale_coords = meta['scale_coords']
        self.scale_class = meta['scale_class']
        self.scale_obj = meta['scale_obj']
        
        self.H, self.W = shape
        
        self.wh = torch.from_numpy(np.reshape([self.W, self.H], [1, 1, 1, 1, 2])).float().cuda()
        anchor_bias_var = torch.from_numpy(np.reshape(self.anchor_bias, [1, 1, 1, self.B, 2])).float().cuda()
        self.anchor_bias_var = anchor_bias_var / self.wh
        self.anchor_padded = torch.cat([torch.zeros((self.B, 2)).cuda(), self.anchor_bias_var.contiguous().view(self.B, 2)], 1)
        
        self.wh_ids = torch.from_numpy(np.stack(np.meshgrid(np.arange(self.W), np.arange(self.H)), 2).reshape((1,self.W,self.H,1,2))).float().cuda()
        
        # functions
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.sigmoid = torch.nn.Sigmoid()

    def loss(self, output, labels, n_truths, early_loss=True):
        predicted = output.permute(0, 2, 3, 1)
        predicted = predicted.contiguous().view(-1, self.H, self.W, self.B, (4 + 1 + self.C))
        batch_size = len(predicted)
        
        #######################################
        # Calculate IOU of boxes
        #####################################
        predicted_xy = self.sigmoid(predicted[..., :2])
        predicted_obj = self.sigmoid(predicted[..., 4:5])

        # # The coordinates are position of cell plus the sigmoid of predicted val
        adjusted_coords = (predicted_xy + self.wh_ids) / self.wh
        # # The width and height are the exponential of predicted multiplied by corresponding anchor
        adjusted_wh = torch.exp(predicted[..., 2:4]) * self.anchor_bias_var
        # # concatenate these two
        xywh = torch.cat([adjusted_coords, adjusted_wh], -1)
        bbox_iou = [bbox_overlap_iou(pred_box, l[:n,1:]) for pred_box,l,n in zip(xywh, labels, n_truths)]

        ##############
        # Calculate the t-values
        #################
        # Calculate the responsible bounding box for each true label
        true_wh = labels[...,3:]
        # this is done by looking at the iou between the label (width and height) with anchor box
        true_iou = bbox_overlap_iou(torch.cat([torch.zeros_like(true_wh), true_wh], -1), self.anchor_padded)
        true_bbox = torch.argmax(true_iou, -1)
        true_anchor = self.anchor_bias_var.squeeze()[true_bbox]
        true_wh_t = (true_wh / true_anchor).log()
        true_ij = labels[...,1:3]*torch.from_numpy(np.array([self.W,self.H])).float().cuda()
        true_xy = true_ij - torch.floor(true_ij)
        true_ij = torch.floor(true_ij).to(torch.long)
        eps = 1e-6
        true_xy_t = torch.log(true_xy + eps) - torch.log(1-true_xy + eps)
        true_coords = torch.cat([true_xy_t, true_wh_t], -1)
        true_coords = torch.cat([c[:n] for c,n in zip(true_coords, n_truths)])
        ######################

        #######################
        # Get indices of the true labels and create a mask from it
        ###################
        true_ijb = torch.cat([true_ij, true_bbox.unsqueeze(-1)], -1)
        n_idx = torch.LongTensor(np.repeat(np.arange(batch_size), n_truths.numpy())).cuda()
        ijb_idx = torch.cat([true_ijb[i, :n_truths[i]] for i in range(batch_size)])
        idx = torch.cat([n_idx.unsqueeze(-1), ijb_idx], -1)
        idx[:,[1,2]] = idx[:,[2,1]] # swap ij axes
        idx = idx.t().chunk(chunks=4,dim=0)
        idx = [id_.squeeze() for id_ in idx]
        mask = torch.zeros((batch_size, self.H, self.W, self.B), dtype=torch.uint8)
        mask[idx] = 1
        ######################

        n_truths_expand = torch.cat([torch.arange(n) for n in n_truths])
        true_ious = [bbox_iou[a][b,c,d,e] for a,b,c,d,e in zip(*idx+[n_truths_expand])]
        sigmas = predicted_obj[idx].squeeze()
        obj_loss = ((torch.stack(true_ious) - sigmas.squeeze())**2).sum()

        coord_loss = ((predicted[idx][:,:4] - true_coords)**2).sum()

        label_idx = torch.cat([lab[:n_truth,0] for n_truth,lab in zip(n_truths, labels)]).to(torch.long)
        logits = predicted[idx][:,5:]
        class_labels = torch.zeros_like(logits)
        class_labels[torch.arange(len(label_idx)),label_idx] = 1
        classification_loss = self.BCEloss(class_labels,logits)

        loss = self.scale_obj * obj_loss + self.scale_coords * coord_loss + self.scale_class * classification_loss
            
        return loss