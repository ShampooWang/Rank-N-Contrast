import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from exp_utils import get_spherical_coordinates
from utils import set_seed
import time

class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2', delta=0.0):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.delta = delta
        if self.delta > 0.0:
            print(f"Adding constraint {self.delta} to the negatives")

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        # Get the ranks of the features by their relative label abosolute differences
        if self.delta > 0.0:
            with torch.no_grad():
                ranks = torch.zeros_like(label_diffs) # [n, n-1]
                for i in range(n):
                    ranks[i] = torch.unique(label_diffs[i], sorted=True, return_inverse=True)[-1]
                margins = (ranks[:, None, :] - ranks[:, :, None]).abs() * self.delta # [n, n-1, n-1]
                del ranks
            logits += logits_max.detach()
            exp_logits = (logits[:, None, :] - margins).exp()

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            if self.delta > 0.0:
                neg_logits = torch.log((neg_mask * exp_logits[:, k, :]).sum(dim=-1))  # 2bs
            else:
                neg_logits = torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            pos_log_probs = pos_logits - neg_logits  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss
    

class CorrelationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: 
        target = target.detach()
        covaraince = ((input - input.mean(0)) * (target - target.mean(0))).mean(0)
        correlation = covaraince / (input.var(dim=0, unbiased=False).sqrt() * target.var(dim=0, unbiased=False).sqrt())

        return -1 * correlation


class PointwiseRankingLoss(nn.Module):
    def __init__(self, feature_norm = "l1", objective = "l1", dim_in: int = 512, data_csv_path: str ='./data/agedb.csv'):
        super().__init__()

        assert feature_norm in ["l1", "l2"], feature_norm
        self.feature_norm = feature_norm

        self.objective = objective
        if self.objective == "l1":
            self.obj_func = nn.L1Loss()
            # self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        elif self.objective == "l2":
            self.obj_func = nn.MSELoss()
        elif self.objective == "covariance":
            self.obj_func = CorrelationLoss(return_covariance=True)
        elif self.objective == "correlation":
            self.obj_func = CorrelationLoss(return_covariance=False)
        elif self.objective == "ordinal":
            import pandas as pd
            y_ranges = torch.LongTensor(sorted(set(pd.read_csv(data_csv_path)["age"]))) # size: [K]
            y_to_order_map = torch.full([y_ranges.max().item()+1], fill_value=-1,)
            y_to_order_map[y_ranges] = torch.arange(len(y_ranges))
            self.thresholds = nn.Parameter(torch.arange(len(y_ranges)-1) + 0.5, requires_grad=False)
            self.y_to_order_map = nn.Parameter(y_to_order_map ,requires_grad=False)
            self.anchor = nn.Parameter(torch.from_numpy(get_spherical_coordinates(dim_in)), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            raise NotImplementedError(objective)
        
    def compute_threshold_loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.detach()
        scores = torch.matmul(features.double(), self.anchor.unsqueeze(1)) # size [2B, 1]
        target_orders = self.y_to_order_map[target.long()] # [2B, 1]
        assert not (target_orders == -1).any(), target_orders # Check whether there are invalid y values
        signs = (target_orders <= self.thresholds.unsqueeze(0)) * 2 - 1 # [2B, K-1]
        loss = (1 + (-1 * signs * (self.thresholds.unsqueeze(0) + self.bias - scores)).exp()).log()

        return loss.sum(1).mean(0)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]
        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        if self.feature_norm == "l1":
            feature_norms = features.norm(dim=-1, p=1) 
        else:
            feature_norms = features.norm(dim=-1, p=2)

        if self.objective == "ordinal":
            # return self.compute_threshold_loss(feature_norms.unsqueeze(-1), labels)
            return self.compute_threshold_loss(features, labels)
        else:
            return self.obj_func(feature_norms.unsqueeze(-1), labels + self.bias.exp())
    

class PairwiseRankingLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2', objective="l2"):
        super(PairwiseRankingLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.objective = objective

        if self.objective == "l1":
            self.obj_func = nn.L1Loss()
        elif self.objective == "l2":
            self.obj_func = nn.MSELoss()
        elif self.objective == "correlation":
            self.obj_func = CorrelationLoss()

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]
        
        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]
        N = features.shape[0]
        device = features.device

        label_diffs = self.label_diff_fn(labels)
        logits = -1 * self.feature_sim_fn(features).div(self.t)
        
        # remove the lower triangular part
        triu_mask = torch.ones((N,N), device=device).triu(diagonal=1).bool()
        masked_logits = logits.masked_select(triu_mask)
        masked_label_diffs = label_diffs.masked_select(triu_mask)

        return self.obj_func(masked_logits, masked_label_diffs)
    

class DeltaOrderLoss(nn.Module):
    def __init__(self, delta: float = 0.1) -> None:
        super(DeltaOrderLoss, self).__init__()
        assert 0 < delta < 1, f"The valid range of delta is (0,1), you assign delta as {delta}"
        self.delta = delta

    def wo_anchor_forward(self, features: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """Compute the delta-order loss

        Args:
            features (torch.Tensor): input features to compute the loss, size: [bs, 2, feat_dim]
            labels (torch.LongTensor): corresponding labels, size: [bs, label_dim]

        Returns:
            torch.Tensor: the computed loss, size: [1]
        """

        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        z_norms = torch.norm(features, dim=1) # [2bs]
        z_norm_diffs = (z_norms[:, None] - z_norms[None, :]) # [2bs, 2bs]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]
        y_diffs = labels - labels.transpose(1,0) # [2bs, 2bs]

        # Remove diagonal
        N, D = features.shape
        mask = (1 - torch.eye(N).to(features.device)).bool()
        y_diffs = y_diffs.masked_select(mask).view(N, N - 1)
        z_norm_diffs = z_norm_diffs.masked_select(mask).view(N, N - 1)
        abs_norm_diffs = z_norm_diffs.abs()

        # Get the ranks of the features by their relative label abosolute differences
        with torch.no_grad():
            ranks = torch.unique(labels, sorted=True, return_inverse=True)[-1]
            rank_diffs = (ranks - ranks.permute(1,0)).masked_select(mask).view(N, N - 1) # [N, N-1]
            pos_weights = (1 + (abs_norm_diffs - self.delta).exp()).log() # [N, N-1]
            # neg_weights = (rank_diffs.abs().div(self.delta) - z_norm_diffs).sigmoid() # [N, N-1]

        # Compute positive logits
        pos_mask = y_diffs == 0
        pos_logits = (abs_norm_diffs * pos_weights * pos_mask).mean()

        # Compute negative logits
        flipped_signs = rank_diffs.sign()
        flipped_diffs = flipped_signs * z_norm_diffs
        # neg_logits = (flipped_diffs * neg_weights * (~pos_mask)).mean()
        # loss = pos_logits - neg_logits
        neg_logits = ((flipped_diffs - rank_diffs.abs().div(self.delta)).abs() * (~pos_mask)).mean()
        loss = pos_logits + neg_logits

        return loss


    def forward(self, features: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """Compute the delta-order loss

        Args:
            features (torch.Tensor): input features to compute the loss, size: [bs, 2, feat_dim]
            labels (torch.LongTensor): corresponding labels, size: [bs, label_dim]

        Returns:
            torch.Tensor: the computed loss, size: [1]
        """
        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]
        y_diffs = labels - labels.transpose(1,0) # [2bs, 2bs]
        z_dists = (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)   
        
        # Remove diagonal
        N, D = features.shape
        mask = (1 - torch.eye(N).to(features.device)).bool()
        y_diffs = y_diffs.masked_select(mask).view(N, N - 1)
        z_dists = z_dists.masked_select(mask).view(N, N - 1)
        y_abs_diffs = y_diffs.abs()

        # flip z_{i,j} - z_{i,k} by sign(y_{i,j} - y_{i,k})
        flipped_signs = (y_abs_diffs[:, None, :] - y_abs_diffs[:, :, None]).sign() # [N, N-1, N-1]
        z_dist_diffs = (z_dists[:, None, :] - z_dists[:, :, None]) # [N, N-1, N-1]
        flipped_dists_diffs = flipped_signs * z_dist_diffs
        abs_diffs = z_dist_diffs.abs()
        
        with torch.no_grad():
            # Get the ranks of the features by their relative label abosolute differences
            ranks = torch.zeros_like(y_abs_diffs) # [N, N-1]
            for i in range(N):
                ranks[i] = torch.unique(y_abs_diffs[i], sorted=True, return_inverse=True)[-1]
            margins = (ranks[:, None, :] - ranks[:, :, None]).div(self.delta).abs() # [N, N-1, N-1]
            del ranks
            # Get the loss weights of the positives and the negatives
            # pos_weights = (abs_diffs - self.delta).sigmoid()  # (abs_diffs - self.delta).sigmoid() or abs_diffs > self.delta 
            neg_weights =  margins > flipped_dists_diffs # (margins - flipped_dists_diffs).sigmoid() or margins > flipped_dists_diffs
     
        # Compute loss
        # pos_masks = y_abs_diffs[:, None, :] == y_abs_diffs[:, :, None]
        # pos_logits = (abs_diffs * pos_weights * pos_masks).sum()
        # neg_logits = (flipped_dists_diffs * neg_weights * (~pos_masks)).sum()
        # loss = (pos_logits - neg_logits) / (N * (N-1))

        # Compute loss
        # pos_masks = y_abs_diffs[:, None, :] == y_abs_diffs[:, :, None]
        # pos_logits = (abs_diffs * pos_weights * pos_masks).mean()
        # neg_logits = (flipped_dists_diffs * neg_weights * (~pos_masks)).mean()
        # loss = pos_logits - neg_logits

        # Compute loss
        neg_masks = y_abs_diffs[:, None, :] != y_abs_diffs[:, :, None]
        neg_logits = ( (margins - flipped_dists_diffs) * neg_weights * neg_masks).mean()
        loss = -1 * neg_logits


        return loss




if __name__ == "__main__":
    set_seed(322)
    features = torch.rand([256, 2, 512], dtype=float)
    labels = torch.randint(1, 80, [256, 1])
    loss = DeltaOrderLoss(delta=0.1).cuda()
    
    # times = []
    # for i in range(500):
    #     times.append(loss(features.cuda(), labels.cuda()))
    # print(sum(times) / len(times))

    print(loss.forward(features.cuda(), labels.cuda()))