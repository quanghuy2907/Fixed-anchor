# ------------------------------------------------------------- #
# Fixed-anchor                                                  #
# Process raw model output                                      #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn.functional as F

from config.config import TEST_CONFIG

class OutputProcessing():
    def __init__(self, dataset_config, device):
        self.dataset_config = dataset_config
        self.device = device

    def process_output(self, output, i):
        ids = np.arange(len(output[i])).reshape(-1, 1)
        x_ids, y_ids = np.unravel_index(ids, (self.dataset_config['grid_height'], self.dataset_config['grid_width']))
        preds = torch.hstack((output[i], torch.tensor(x_ids, device=self.device), torch.tensor(y_ids, device=self.device)))
        preds = preds[preds[:, 0] >= TEST_CONFIG['conf_thres']]

        final_slots = torch.cat([(preds[:, 14:15] + 0.5) * self.dataset_config['cell_size'] + preds[:, 1:2] * self.dataset_config['l_max'],
                                 (preds[:, 13:14] + 0.5) * self.dataset_config['cell_size'] + preds[:, 2:3] * self.dataset_config['l_max'],
                                 (preds[:, 14:15] + 0.5) * self.dataset_config['cell_size'] + preds[:, 3:4] * self.dataset_config['l_max'],
                                 (preds[:, 13:14] + 0.5) * self.dataset_config['cell_size'] + preds[:, 4:5] * self.dataset_config['l_max'],
                                 F.normalize(preds[:, 5:7]),
                                 F.normalize(preds[:, 7:9]),
                                 torch.argmax(preds[:, 9:12], dim=-1).reshape(-1, 1),
                                 torch.round(preds[:, 12:13])], dim=-1)

        return final_slots
    

    def visualize_preds(self, imgs, final_slots, index):
        img = imgs[index].detach().cpu().permute(1,2,0)
        plt.figure()
        plt.axis('off')
        plt.imshow(img)

        for item in final_slots:
            junc1 = np.array([item[0], item[1]])
            junc2 = np.array([item[2], item[3]])

            ori1 = np.array([junc1[0] + item[4]*25, junc1[1] + item[5]*25])
            ori2 = np.array([junc2[0] + item[6]*25, junc2[1] + item[7]*25])

            slot_type = int(item[8])
            slot_occ = item[9]
            if slot_occ == 0:
                plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
            else:
                plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
        
        # plt.savefig('b.png')
        # plt.close()
        plt.show()


    