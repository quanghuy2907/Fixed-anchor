# ------------------------------------------------------------- #
# Fixed-anchor                                                  #
# Testing process                                               #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
import time

import torch
from torch.utils.data import DataLoader

from dataloader.data_pipeline import CustomDataset
from model.fixed_anchor import Fixed_anchor
from config.config import DATASET, MODEL_CONFIG, TEST_CONFIG
from inference.output_processing import OutputProcessing
from inference.evaluation import Evaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = CustomDataset(dataset=DATASET, state='test')
test_loader = DataLoader(test_dataset, batch_size=TEST_CONFIG['batch_size'], shuffle=False, drop_last=False)

model = Fixed_anchor(d_model=MODEL_CONFIG['d_model'], 
                     nhead=MODEL_CONFIG['n_head'], 
                     dim_feedforward=MODEL_CONFIG['ffn'], 
                     dropout=MODEL_CONFIG['dropout'],
                     num_encoder_layers=MODEL_CONFIG['num_enc_layer'], 
                     num_decoder_layers=MODEL_CONFIG['num_dec_layer'],
                     num_queries=test_dataset.dataset_config['object_num'], 
                     activation=MODEL_CONFIG['activation'], 
                     return_intermediate_dec=MODEL_CONFIG['return_inter'])
model = model.to(device=device)

weight_path = './weights/snu_20250801_213324/model_weights.pth'
model.load_state_dict(torch.load(weight_path, weights_only=True))
model.eval()

output_processing_tool = OutputProcessing(test_dataset.dataset_config, device)
eval_tool = Evaluation(test_dataset.dataset_config, threshold='loose')

inf_time = 0

start_time = time.time()
print('-----Start testing-----')
for idx, data in enumerate(test_loader):
    imgs, labels = data
    imgs = imgs.to(device=device, dtype=torch.float32)

    t1 = time.time()
    output = model(imgs)
    inf_time += (time.time() - t1)

    for i in range(TEST_CONFIG['batch_size']):
        sys.stdout.write('\r' + 'idx=' + str(i + idx*TEST_CONFIG['batch_size'] + 1))

        final_slots = output_processing_tool.process_output(output, i)        
        final_slots = final_slots.detach().cpu().numpy()
        gt_slots = labels[i][labels[i, :, 0] != 0]
        eval_tool.evaluate(final_slots, gt_slots)


print(f'\nTotal testing time: {time.time() - start_time:.4f} secs')

inference_time = inf_time / (len(test_loader) * TEST_CONFIG['batch_size'])
print(f'Inference time: {inference_time} sec/frame')
print(f'Inference time: {1/ inference_time} fps')
print('\n')

eval_tool.results()
