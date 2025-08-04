# ------------------------------------------------------------- #
# Fixed-anchor                                                  #
# Hungarian matcher                                             #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, conf_weight, dist_weight):
        super().__init__()

        self.conf_weight = conf_weight
        self.dist_weight = dist_weight

    def get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def forward(self, preds, trues):
        with torch.no_grad():
            bs, num_queries = preds.shape[:2]
            # We flatten to compute the cost matrices in a batch
            pred_conf = preds[..., 0:1].flatten(0, 1)
            pred_junc1 = preds[..., 1:3].flatten(0, 1)
            pred_junc2 = preds[..., 3:5].flatten(0, 1)

            true_flatten = trues.flatten(0, 1)
            true_flatten = true_flatten[true_flatten[:, 0] != 0]
            true_conf = true_flatten[..., 0:1]
            true_junc1 = true_flatten[..., 1:3]
            true_junc2 = true_flatten[..., 3:5]

            
            # Compute the confidence cost
            cost_conf = torch.cdist(pred_conf, true_conf, p=2)

            # Compute the distance cost
            cost_dist =  torch.cdist(pred_junc1, true_junc1, p=2) + torch.cdist(pred_junc2, true_junc2, p=2)
            
            C = self.conf_weight * cost_conf + self.dist_weight * cost_dist
            C = C.view(bs, num_queries, -1).cpu()

            
            sizes = [torch.sum(item[:, 0] != 0) for item in trues]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

