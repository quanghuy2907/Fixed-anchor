# ------------------------------------------------------------- #
# Fixed-anchor                                                  #
# Input pipeline                                                #
#       - Read images                                           #
#       - Read annotations                                      #
#       - Data augmentation                                     #
#       - Generate labels                                       #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #

import os
import numpy as np
import skimage
import scipy
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config.config import DATASET_CONFIG

class CustomDataset(Dataset):
    def __init__(self, dataset='ps2', state='train'):  
        if dataset not in DATASET_CONFIG:
            raise ValueError(f'Unknown dataset: {dataset}')
        
        self.dataset = dataset
        self.dataset_config = DATASET_CONFIG[dataset]

        self.img_path = os.path.join(self.dataset_config['dataset_path'], 'images', state if state == 'train' else 'test')
        self.label_path = os.path.join(self.dataset_config['dataset_path'], 'labels', state if state == 'train' else 'test')
        self.img_files = os.listdir(self.img_path)
        self.img_files.sort()

        self.state = state



    def __len__(self):
        return len(self.img_files)
    


    # Extract slot information from raw .mat data file 
    def read_slot_annotation(self, file_path):
        data = scipy.io.loadmat(file_path)

        temp_junctions = []
        temp_slots = []
        slots = []

        if self.dataset == 'ps2':
            for i in range(len(data['marks'])):
                temp_junctions.append([float(data['marks'][i][0]), float(data['marks'][i][1]), 
                                    float(data['marks'][i][2]), float(data['marks'][i][3])])
            for i in range(len(data['slots'])):  
                temp_slots.append([int(data['slots'][i][0]), int(data['slots'][i][1]), 
                                int(data['slots'][i][7] - 1),
                                int(data['slots'][i][8])])
            for slot in temp_slots:
                slot_type = slot[2]
                slot_occ = slot[3]

                point1 = [temp_junctions[int(slot[0] - 1)][0], temp_junctions[int(slot[0] - 1)][1]]
                ori1 = [temp_junctions[int(slot[0] - 1)][2], temp_junctions[int(slot[0] - 1)][3]]
                
                point2 = [temp_junctions[int(slot[1] - 1)][0], temp_junctions[int(slot[1] - 1)][1]]
                ori2 = [temp_junctions[int(slot[1] - 1)][2], temp_junctions[int(slot[1] - 1)][3]]

                slots.append([point1[0], point1[1], point2[0], point2[1],
                              ori1[0], ori1[1], ori2[0], ori2[1], 
                              slot_type, slot_occ])
        else: # SNU
            for i in range(len(data['marks'])):
                temp_junctions.append([float(data['marks'][i][0]), float(data['marks'][i][1])])
            for i in range(len(data['slots'])):  
                temp_slots.append([int(data['slots'][i][0]), int(data['slots'][i][1]), 
                                int(data['slots'][i][2]), int(data['slots'][i][3]),
                                int(data['slots'][i][4]),
                                int(data['slots'][i][5])])
            for slot in temp_slots:
                slot_type = slot[4]
                slot_occ = slot[5]

                point1 = np.array([temp_junctions[int(slot[0] - 1)][0], temp_junctions[int(slot[0] - 1)][1]])
                point2 = np.array([temp_junctions[int(slot[1] - 1)][0], temp_junctions[int(slot[1] - 1)][1]])
                point3 = np.array([temp_junctions[int(slot[2] - 1)][0], temp_junctions[int(slot[2] - 1)][1]])
                point4 = np.array([temp_junctions[int(slot[3] - 1)][0], temp_junctions[int(slot[3] - 1)][1]])

                ori1 = point4 - point1
                ori1 = ori1 / (np.linalg.norm(ori1) + 1e-9)
                ori2 = point3 - point2
                ori2 = ori2 / (np.linalg.norm(ori2) + 1e-9)
                if np.linalg.norm(ori1) == 0:
                    ori1 = ori2
                if np.linalg.norm(ori2) == 0:
                    ori2 = ori1
                if np.linalg.norm(ori1) == 0 and np.linalg.norm(ori2) == 0:
                    print(f'There is a slot with error label in {file_path}')
                    continue

                slots.append([point1[0], point1[1], point2[0], point2[1],
                              ori1[0], ori1[1], ori2[0], ori2[1],
                              slot_type, slot_occ])

        slots = np.array(slots)
        if len(slots) > 0:
            slots[:, 0:4] *= self.dataset_config['ratio']

        return slots
    


    # Implement aumentation for the input (include horizontal and vertical flip)
    def augment(self, img, slots):
        aug_img = img.copy()
        aug_slots = slots.copy()

        # Vertical flip
        if np.random.random() > 0.5:
            aug_img = cv2.flip(aug_img, 0)
            temp = []
            for slot in aug_slots:
                temp.append([slot[2], self.dataset_config['img_height'] - slot[3],
                             slot[0], self.dataset_config['img_height'] - slot[1],
                             slot[6], -slot[7],
                             slot[4], -slot[5],
                             slot[8],
                             slot[9]])
            aug_slots = np.array(temp)
        
        # Horizontal flip (only work for PS2.0 dataset)
        if self.dataset_config['flip_horizontal']:
            if np.random.random() > 0.5:
                aug_img = cv2.flip(aug_img, 1)
                temp = []
                for slot in aug_slots:
                    temp.append([self.dataset_config['img_width'] - slot[2], slot[3],
                                 self.dataset_config['img_width'] - slot[0], slot[1],
                                 -slot[6], slot[7],
                                 -slot[4], slot[5],
                                 slot[8],
                                 slot[9]])
                aug_slots = np.array(temp)
            
        return aug_img, aug_slots



    # Generate label for training
    def generate_slot_labels(self, slots, state):
        if state == 'train' or state == 'val':
            labels = np.zeros((self.dataset_config['grid_height'], self.dataset_config['grid_width'], self.dataset_config['label_channel']))

            for slot in slots:
                entrance_center = ((slot[0:2] + slot[2:4]) / 2)
                xid = int(entrance_center[1] / self.dataset_config['cell_size'])
                yid = int(entrance_center[0] / self.dataset_config['cell_size'])
                
                # Object score 
                labels[xid, yid, 0] = 1
                # Junction locations
                labels[xid, yid, 1:3] = (slot[0:2] - np.array([(yid + 0.5) * self.dataset_config['cell_size'], (xid + 0.5) * self.dataset_config['cell_size']])) / self.dataset_config['l_max']
                labels[xid, yid, 3:5] = (slot[2:4] - np.array([(yid + 0.5) * self.dataset_config['cell_size'], (xid + 0.5) * self.dataset_config['cell_size']])) / self.dataset_config['l_max']
                # Junction orientations
                ori1 = np.array([slot[4], slot[5]])
                ori2 = np.array([slot[6], slot[7]])
                if np.linalg.norm(ori1) == 0:
                    ori1 = ori2
                if np.linalg.norm(ori2) == 0:
                    ori2 = ori1
                ori1 = ori1/np.linalg.norm(ori1)
                ori2 = ori2/np.linalg.norm(ori2)

                labels[xid, yid, 5:9] = [ori1[0], ori1[1], ori2[0], ori2[1]]

                # Slot type
                slot_type = int(slot[8])
                labels[xid, yid, 9 + slot_type] = 1
                # Slot occupancy
                slot_occ = int(slot[9])
                labels[xid, yid, 12] = slot_occ

            labels = labels.reshape([-1, self.dataset_config['label_channel']])
        
        elif state == 'test':
            labels = slots
            if len(labels) != 0:
                labels = np.pad(labels, pad_width=((0, 10 - len(slots)), (0, 0)), mode='constant', constant_values=0)
            else:
                labels = np.zeros((10, 10))
                
        return labels


    # Visualization function, for testing
    def visualize_label(self, img, label):
        plt.figure()
        plt.imshow(img.permute(1, 2, 0))

        label = label.numpy()
        if self.state == 'train':
            for id, item in enumerate(label):
                if item[0] != 0:
                    xid, yid = np.unravel_index(id, (self.dataset_config['grid_height'], self.dataset_config['grid_width']))
                    junc1 = np.array([yid + 0.5, xid + 0.5]) * self.dataset_config['cell_size'] + item[1:3] * self.dataset_config['l_max']
                    junc2 = np.array([yid + 0.5, xid + 0.5]) * self.dataset_config['cell_size'] + item[3:5] * self.dataset_config['l_max']

                    ori1 = np.array([junc1[0] + item[5]*25, junc1[1] + item[6]*25])
                    ori2 = np.array([junc2[0] + item[7]*25, junc2[1] + item[8]*25])

                    slot_type = np.argmax(item[9:12])
                    slot_occ = item[12]
                    if slot_occ == 0:
                        plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                        plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                        plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '-', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                    else:
                        plt.plot([junc1[0], junc2[0]], [junc1[1], junc2[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                        plt.plot([junc1[0], ori1[0]], [junc1[1], ori1[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)
                        plt.plot([junc2[0], ori2[0]], [junc2[1], ori2[1]], '--', color=self.dataset_config['color_dict'][slot_type], linewidth=3)

        elif self.state == 'test':
            for item in label:
                if np.sum(item) != 0:
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

        plt.show()


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.img_files[idx])
        image = skimage.io.imread(img_path)
        image = np.uint8(np.round(skimage.transform.resize(image, (self.dataset_config['img_height'], self.dataset_config['img_width']), preserve_range=True)))

        label_path = os.path.join(self.label_path, self.img_files[idx].replace('jpg', 'mat'))
        slots = self.read_slot_annotation(label_path)

        # Augmentation (flip horizontal + vertical)
        if self.state == 'train':
            aug_image, aug_slots = self.augment(image, slots)
        else:
            aug_image = image.copy()
            aug_slots = slots.copy()

        aug_image = transforms.ToTensor()(aug_image)
        labels = self.generate_slot_labels(aug_slots, self.state)

        return aug_image, labels
    