# ------------------------------------------------------------- #
# Fixed-anchor                                                  #
# Configiguration file                                          #
#       - Dataset configurations (SNU, PS2.0)                   #
#       - Model configurations                                  #
#       - Training/Testing set up                               #
#       - Evaluation criteria                                   #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


WEIGHT_DIR = './weights/'
LOG_DIR = './log/'

DATASET = 'snu'
DATASET_CONFIG = {
    'ps2': {
        'dataset_path': '/home/ivpg/HUY/Dataset/PS2/',
        'ori_height': 600,
        'ori_width': 600,
        'img_height': 416,
        'img_width': 416,
        'ratio': 416/600, # img_height / ori_height
        'l_max': 7.0 * (416 / 10.0), # 7m * (img_height / 10m)

        'conf_weights': [1.0401, 25.9260], # [not contain (0): 236199, contain (1): 9476]
        'class_name': ['perpendicular', 'parallel', 'slanted'],
        'color_dict': ['g', 'r', 'b'],
        'class_weights': [1.6718, 2.7136, 29.9873], #[per: 5668, par: 3492, sla: 316]
        'occupancy_weights': [1, 1],
        'flip_horizontal': True,    

        'grid_height': 5,
        'grid_width': 5,
        'cell_size': 416/5, # img_height / grid_height
        'object_num': 5*5, # grid_height * grid_width
        'label_channel': 13,
    },
    'snu': { 
        'dataset_path': '/home/ivpg/HUY/Dataset/SNU/',
        'ori_height': 768,
        'ori_width': 256,
        'img_height': 576,
        'img_width': 192,
        'ratio': 576/768, # img_height / ori_height
        'l_max': 400 * (576/768), # 400pixel * ratio

        'conf_weights': [1.1171, 9.5427], # [not contain (0): 442298, contain (1): 51775]
        'class_name': ['parallel', 'perpendicular', 'slanted'],
        'color_dict': ['r', 'g', 'b'],
        'class_weights': [8.2065, 1.2331, 14.8779], #[par: 6309, per: 41986, sla: 3480]
        'occupancy_weights': [1.7136, 2.4013], #[vac (0): 30214, occ (1): 21561]
        'flip_horizontal': False,

        'grid_height': 9,
        'grid_width': 3,
        'cell_size': 576/9, # img_height / grid_height
        'object_num': 9*3, # grid_height * grid_width
        'label_channel': 13,
    }
}


MODEL_CONFIG = {
    'd_model': 256,
    'n_head': 8,
    'ffn': 1024,
    'dropout': 0.1,
    'num_enc_layer': 6,
    'num_dec_layer': 6,
    'activation': 'relu',
    'return_inter': False,
}


LOSS_CONFIG = {
    'conf': 10,
    'loc': 1000,
    'ori': 1000,
    'type': 1,
    'occ': 10,
}


TRAIN_CONFIG = {
    'batch_size': 32,
    'epoch_num': 200,
    'lr_init': 1e-4,
    'lr_end': 1e-6,
}


TEST_CONFIG = {
    'batch_size': 1,
    'conf_thres': 0.7,
    'loose_loc_thres': 12, 
    'loose_ori_thres': 10,
    'tight_loc_thres': 6,
    'tight_ori_thres': 5,
}

