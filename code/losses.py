import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, image_logits, text_logits):
        labels = torch.arange(image_logits.shape[0]).to(image_logits.device)
        image_loss = F.cross_entropy(image_logits, labels)
        text_loss = F.cross_entropy(text_logits, labels)
        return (image_loss + text_loss) / 2

class RankingLoss(nn.Module):
    def __init__(self, args):
        super(RankingLoss, self).__init__()

        self.scale = args.scale
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.num_trans = args.num_trans
        self.ranking_batch = args.ranking_batch

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=2, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        sim_matrix = torch.einsum('bij,bjk->bik', text_features.unsqueeze(1), 
                                                   image_features.permute(1,2,0)).squeeze()

        
        #Compute listwise ranking loss
        diff_matrix = torch.add((sim_matrix[:,1:] - sim_matrix[:,:self.num_trans]), self.alpha)
        scaled_diff_matrix = torch.exp(torch.mul(diff_matrix, self.scale))
        sum_matrix = torch.add(torch.sum(scaled_diff_matrix, dim=1), 1.0)
        scaled_log_matrix = torch.div(torch.log(sum_matrix), self.scale)
        list_loss = torch.div(torch.sum(scaled_log_matrix), self.ranking_batch)

        #Compute positive loss
        negative_sim = torch.add(torch.mul(sim_matrix, -1.0), self.beta)
        scaled_pos_matrix = torch.exp(torch.mul(negative_sim, self.scale))
        sum_pos_matrix = torch.add(torch.sum(scaled_pos_matrix, dim=1), 1.0)
        scaled_log_pos_matrix = torch.div(torch.log(sum_pos_matrix), self.scale)
        pos_loss = torch.div(torch.sum(scaled_log_pos_matrix), self.ranking_batch)

        return list_loss + self.gamma * pos_loss
