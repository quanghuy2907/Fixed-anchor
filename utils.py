# ------------------------------------------------------------- #
# Fixed-anchor                                                  #
# Util functions                                                #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import numpy as np
import math
import matplotlib.pyplot as plt



def distance(point1, point2):
    dist = np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))
    return dist

def diff_angle(ori1, ori2):
    ori1 = ori1/np.linalg.norm(ori1)
    ori2 = ori2/np.linalg.norm(ori2)
    angle = np.arccos(max(min(1.0, np.dot(ori1, ori2)), -1.0))
    angle = angle*180/math.pi
    return angle





def visualize_preds(imgs, final_slots, dataset_config, index):
    img = imgs[index].detach().cpu().permute(1,2,0)
    plt.figure()
    plt.imshow(img)

    for item in final_slots:
        junc1 = np.array([item[0], item[1]])
        junc2 = np.array([item[2], item[3]])

        ori1 = np.array([junc1[0] + item[4]*25, junc1[1] + item[5]*25])
        ori2 = np.array([junc2[0] + item[6]*25, junc2[1] + item[7]*25])

        slot_type = int(8)
        slot_occ = item[9]
        if slot_occ == 0:
            plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '-', color=dataset_config['color_dict'][slot_type], linewidth=3)
            plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '-', color=dataset_config['color_dict'][slot_type], linewidth=3)
            plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '-', color=dataset_config['color_dict'][slot_type], linewidth=3)
        else:
            plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '--', color=dataset_config['color_dict'][slot_type], linewidth=3)
            plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '--', color=dataset_config['color_dict'][slot_type], linewidth=3)
            plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '--', color=dataset_config['color_dict'][slot_type], linewidth=3)
    
    # plt.savefig('b.png')
    # plt.close()
    plt.show()


