# ------------------------------------------------------------- #
# Fixed-anchor                                                  #
# Loss funtions                                                 #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.aggregation import MeanMetric

from config.config import LOSS_CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Metric():
    def __init__(self, log_folder):
        self.conf = MeanMetric().to(device)
        self.loc1 = MeanMetric().to(device)
        self.loc2 = MeanMetric().to(device)
        self.ori1 = MeanMetric().to(device)
        self.ori2 = MeanMetric().to(device)
        self.type = MeanMetric().to(device)
        self.occ = MeanMetric().to(device)
        self.total_loss = MeanMetric().to(device)

        self.writer = SummaryWriter(log_folder)
    
    def reset(self):
        self.conf.reset()
        self.loc1.reset()
        self.loc2.reset()
        self.ori1.reset()
        self.ori2.reset()
        self.type.reset()
        self.occ.reset()
        self.total_loss.reset()

    def update(self, losses):
        self.conf.update(losses[0])
        self.loc1.update(losses[1])
        self.loc2.update(losses[2])
        self.ori1.update(losses[3])
        self.ori2.update(losses[4])
        self.type.update(losses[5])
        self.occ.update(losses[6])
        self.total_loss.update(losses[7])
    
    def write(self, epoch, lr, state):
        if state == 'train':
            self.writer.add_scalar('lr', lr, global_step=epoch)

        self.writer.add_scalar(state + '_loss/conf_loss', self.conf.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/loc1_loss', self.loc1.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/loc2_loss', self.loc2.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/ori1_loss', self.ori1.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/ori2_loss', self.ori2.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/slot_type_loss', self.type.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/slot_occ_loss', self.occ.compute(), global_step=epoch)
        self.writer.add_scalar(state + '_loss/total_loss', self.total_loss.compute(), global_step=epoch)

        self.writer.flush()

    def compute(self):
        return [self.conf.compute(),
                self.loc1.compute(),
                self.loc2.compute(),
                self.ori1.compute(),
                self.ori2.compute(),
                self.type.compute(),
                self.occ.compute(),
                self.total_loss.compute()]
        



class Losses():
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config


    def conf_loss(self, pred, true, weight):
        loss = torch.square(pred - true)
        loss = (true == 0) * loss * weight[0] + (true == 1) * loss * weight[1]
        # loss = torch.sum(loss, dim=[1, 2])
        loss = torch.mean(loss) 

        return loss
    

    def loc_loss(self, pred, true, mask):
        loss = torch.square(pred - true)
        loss = mask * torch.sum(loss, -1)
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-9)

        return loss
    
    def ori_loss(self, pred, true, mask):
        pred = F.normalize(pred, dim=-1)
        true = F.normalize(true, dim=-1)

        loss = torch.square(pred - true)
        loss = mask * torch.sum(loss, -1)
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-9)

        return loss
    
    def type_loss(self, pred, true, mask, weight):
        pred = pred.permute(0, 2, 1)
        true = torch.argmax(true, dim=-1).to(torch.long)
        class_weights = torch.tensor(weight).to(pred.device)

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        loss = loss_fn(pred, true)
        loss = mask * loss
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-9)

        return loss
    
    def occ_loss(self, pred, true, mask, weight):
        loss = torch.square(pred - true)
        loss = mask * loss
        loss = (true == 0) * loss * weight[0] + (true == 1) * loss * weight[1]
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-9)

        return loss



    def compute(self, output, labels):
        conf = self.conf_loss(output[:, :, 0], labels[:, :, 0], self.dataset_config['conf_weights']) * LOSS_CONFIG['conf']
        loc1 = self.loc_loss(output[:, :, 1:3], labels[:, :, 1:3], labels[:, :, 0] == 1) * LOSS_CONFIG['loc']
        loc2 = self.loc_loss(output[:, :, 3:5], labels[:, :, 3:5], labels[:, :, 0] == 1) * LOSS_CONFIG['loc']
        ori1 = self.ori_loss(output[:, :, 5:7], labels[:, :, 5:7], labels[:, :, 0] == 1) * LOSS_CONFIG['ori']
        ori2 = self.ori_loss(output[:, :, 7:9], labels[:, :, 7:9], labels[:, :, 0] == 1) * LOSS_CONFIG['ori']
        slot_type = self.type_loss(output[:, :, 9:12], labels[:, :, 9:12], labels[:, :, 0] == 1, self.dataset_config['class_weights']) * LOSS_CONFIG['type']
        slot_occ = self.occ_loss(output[:, :, 12], labels[:, :, 12], labels[:, :, 0] == 1, self.dataset_config['occupancy_weights']) * LOSS_CONFIG['occ']

        total = conf + loc1 + loc2 + ori1 + ori2 + slot_type + slot_occ

        return [conf, loc1, loc2, ori1, ori2, slot_type, slot_occ, total]


