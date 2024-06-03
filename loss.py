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
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels, ranks):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]
        if features.ndim == 3:
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

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss
    

class CorrelationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: 
        target = target.detach()
        covaraince = ((input - input.mean(0)) * (target - target.mean(0))).mean(0)
        correlation = covaraince.div(input.std(dim=0, unbiased=False) * target.std(dim=0, unbiased=False))

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

    def forward(self, features, labels, ranks):
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
    def __init__(self, delta=1.0, objective="l1", eps=0.0):
        super(PairwiseRankingLoss, self).__init__()
        self.delta = delta
        self.objective = objective
        self.eps = eps

        if self.objective == "l1":
            self.obj_func = nn.L1Loss(reduction="none")
        elif self.objective == "l2":
            self.obj_func = nn.MSELoss(reduction="none")
        elif self.objective == "huber":
            self.obj_func = nn.HuberLoss(reduction="none")

    def forward(self, features: torch.Tensor, labels: torch.Tensor, ranks: torch.Tensor):
        """Compute the pairwiseranking loss

        Args:
            features (torch.Tensor): input features to compute the loss, size: [bs, 2, feat_dim]
            labels (torch.LongTensor): corresponding labels, size: [bs, label_dim]
            ranks (torch.LongTensor): corresponding labels' ranks, size: [bs, label_dim]
        Returns:
            torch.Tensor: the computed loss, size: [1]
        """
        
        if features.ndim == 3:
            features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
            ranks = ranks.repeat(2, 1)  # [2bs, label_dim]

        z_dists = (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)   
        rank_diffs = ranks - ranks.transpose(1,0)
        N = features.shape[0]
        device = features.device
        
        # remove the lower triangular part
        # diagonal_mask = (1 - torch.eye(N, device=device)).bool()
        triu_mask = torch.ones((N,N), device=device).triu(diagonal=1).bool()
        z_dists = z_dists.masked_select(triu_mask)
        rank_diffs = rank_diffs.masked_select(triu_mask)

        loss = self.obj_func(z_dists, self.delta * rank_diffs.abs())
        
        if self.eps > 0.0:
            return torch.where(loss <= self.eps, 0, loss).mean()
        else:
            return loss.mean()
    

class DeltaOrderLoss(nn.Module):
    def __init__(self, delta: float = 0.1, objective="l2") -> None:
        super(DeltaOrderLoss, self).__init__()
        # assert 0 < delta < 1, f"The valid range of delta is (0,1), you assign delta as {delta}"
        self.delta = delta
        self.weighted_func = nn.Softplus()
        self.objective = objective
        if self.objective == "l1":
            self.obj_func = nn.L1Loss(reduction='none')
        elif self.objective == "l2":
            self.obj_func = nn.MSELoss(reduction='none')
        elif self.objective == "huber":
            self.obj_func = nn.HuberLoss(reduction='none')

    def wo_anchor_forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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

    def forward(self, features: torch.Tensor, labels: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
        """Compute the delta-order loss

        Args:
            features (torch.Tensor): input features to compute the loss, size: [bs, 2, feat_dim]
            labels (torch.Tensor): corresponding labels, size: [bs, label_dim]

        Returns:
            torch.Tensor: the computed loss, size: [1]
        """
        if features.ndim == 3:
            features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
            labels = labels.repeat(2, 1)  # [2bs, label_dim]
            # ranks = ranks.repeat(2, 1)

        y_diffs = labels - labels.transpose(1,0) # [2bs, 2bs]
        z_dists = (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)   
        
        # Remove diagonal
        N, D = features.shape
        mask = (1 - torch.eye(N).to(features.device)).bool()
        y_diffs = y_diffs.masked_select(mask).view(N, N - 1)
        z_dists = z_dists.masked_select(mask).view(N, N - 1)
        y_abs_diffs = y_diffs.abs()

        # flip z_{i,j} - z_{i,k} by sign(y_{i,j} - y_{i,k})
        # flipped_signs = (y_abs_diffs[:, None, :] - y_abs_diffs[:, :, None]).sign() # [N, N-1, N-1]
        z_dist_diffs = (z_dists[:, None, :] - z_dists[:, :, None]) # [N, N-1, N-1]
        # flipped_dists_diffs = flipped_signs * z_dist_diffs
        abs_diffs = z_dist_diffs.abs()
        
        with torch.no_grad():
            # Get the ranks of the features by their relative label abosolute differences
            ranks = torch.zeros_like(y_abs_diffs) # [N, N-1]
            for i in range(N):
                ranks[i] = torch.unique(y_abs_diffs[i], sorted=True, return_inverse=True)[-1]
            margins = (ranks[:, None, :] - ranks[:, :, None]).abs() * self.delta # [N, N-1, N-1]
            del ranks
            # Get the loss weights of the positives and the negatives
            pos_weights = (abs_diffs - self.delta).sigmoid()   # (abs_diffs - self.delta).sigmoid() or abs_diffs > self.delta 
            # neg_weights =  margins > flipped_dists_diffs # (margins - flipped_dists_diffs).sigmoid() or margins > flipped_dists_diffs
     
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
        pos_masks = y_abs_diffs[:, None, :] == y_abs_diffs[:, :, None]
        # pos_weights = self.weighted_func(abs_diffs - self.delta)
        pos_logits = (abs_diffs * pos_weights * pos_masks).mean()
        neg_logits = (self.obj_func(abs_diffs, margins) * (~pos_masks)).mean()
        loss = pos_logits + neg_logits


        return loss


class ProbRankingLoss(nn.Module):
    def __init__(self, t: float=2.0):
        super(ProbRankingLoss, self).__init__()
        self.t = t
        # self.std = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.softmax = nn.Softmax(dim=1)
        # self.softplus = nn.Softplus()
        self.obj_func = nn.KLDivLoss(reduction="batchmean")
        self.coeff = math.sqrt(2*math.pi)
        
    def compute_normal_probs(self, logits, t=1.0):
        std = logits.norm(dim=1, keepdim=True).div(t)
        probs = (-0.5 * (logits/std).square()).exp().div(self.coeff * std)

        return probs

    def forward(self, features: torch.Tensor, labels: torch.Tensor, ranks: torch.Tensor):
        """Compute the pairwiseranking loss

        Args:
            features (torch.Tensor): input features to compute the loss, size: [bs, 2, feat_dim]
            labels (torch.LongTensor): corresponding labels, size: [bs, label_dim]
            ranks (torch.LongTensor): corresponding labels' ranks, size: [bs, label_dim]
        Returns:
            torch.Tensor: the computed loss, size: [1]
        """
        
        if features.ndim == 3:
            features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
            labels = labels.repeat(2, 1)
            # ranks = ranks.repeat(2, 1)  # [2bs, label_dim]

        z_dists = (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)   
        y_diffs = labels - labels.transpose(1,0) # [2bs, 2bs]
        # rank_diffs = ranks - ranks.transpose(1,0)
        N = features.shape[0]
        
        # remove diagonal
        mask = (1 - torch.eye(N).to(features.device)).bool()
        y_diffs = y_diffs.masked_select(mask).view(N, N - 1)
        y_abs_diffs = y_diffs.abs()
        z_dists = z_dists.masked_select(mask).view(N, N - 1)
        log_z_probs = F.log_softmax(-z_dists.div(self.t), dim=1)
        # rank_diffs = rank_diffs.masked_select(mask).view(N, N - 1)

        # flip z_{i,j} - z_{i,k} by sign(y_{i,j} - y_{i,k})
        # flipped_signs = (y_diffs[:, None, :] - y_diffs[:, :, None]).sign() # [N, N-1, N-1]
        # z_dist_diffs = (z_dists[:, None, :] - z_dists[:, :, None]) # [N, N-1, N-1]
        # flipped_dists_diffs = flipped_signs * z_dist_diffs
        # abs_diffs = z_dist_diffs.abs()

        # with torch.no_grad():
            # Get the ranks of the features by their relative label abosolute differences
            # ranks = torch.zeros_like(y_abs_diffs) # [N, N-1]
            # for i in range(N):
            #     ranks[i] = torch.unique(y_abs_diffs[i], sorted=True, return_inverse=True)[-1]
            # ranks_dists = (ranks[:, None, :] - ranks[:, :, None]).abs()
            # ordinary_masks = (ranks[:, None, :] - 1)  == ranks[:, :, None]

        # logits = (z_dists[:, None, :] - z_dists[:, :, None]).exp()
        # loss = 0.0
        # for j in range(N-1):
        #     probs_diffs_log = (1 + z_probs[:, j, None] - z_probs).log()
        #     mask = (ranks[:, j, None] + 1) == ranks
        #     loss = -(probs_diffs_log * mask).sum()

        return self.obj_func(log_z_probs, F.softmax(-y_abs_diffs.div(self.t), dim=1))


class kNNRnCLoss(nn.Module):
    def __init__(self, k=10, temperature=2, label_diff='l1', feature_sim='l2'):
        super(kNNRnCLoss, self).__init__()
        self.k = k
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels, ranks):
        if features.ndim == 3:
            # features: [bs, 2, feat_dim]
            # labels: [bs, label_dim]
            features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
            labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = (features[:, None, :] - features[None, :, :]).norm(2, dim=-1).div(self.t) # (features[:, None, :] - features[None, :, :]).norm(2, dim=-1).div(self.t), self.feature_sim_fn(features).div(self.t)
        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        diag_mask = (1 - torch.eye(n).to(logits.device)).bool()
        logits = logits.masked_select(diag_mask).view(n, n - 1)
        label_diffs = label_diffs.masked_select(diag_mask).view(n, n - 1)

        # Sort logits by the label_diffs
        # with torch.no_grad():
        sorted_labdiffs, sorted_indices = torch.sort(label_diffs, dim=1)
        sorted_logits = torch.gather(logits, 1, sorted_indices)
        
        # exp_logits = sorted_logits.exp()
        # logits_max = sorted_logits[:, -1]
        # sorted_logits -= logits_max[:, None].detach()
        # loss = 0.
        # pos_num = min(self.k, n-1)
        # for k in range(pos_num):
        #     pos_logits = sorted_logits[:, k]  # 2bs
        #     pos_label_diffs = sorted_labdiffs[:, k]  # 2bs
        #     neg_mask = (sorted_labdiffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
        #     pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
        #     loss += - (pos_log_probs / (pos_num * (pos_num - 1))).sum()

        loss = 0.
        pos_num = min(self.k, n-1)
        for k in range(pos_num):
            pos_logits = sorted_logits[:, k]  # 2bs
            logits_diffs = sorted_logits - pos_logits[:, None]
            pos_label_diffs = sorted_labdiffs[:, k]  # 2bs
            flipped_signs = torch.where(sorted_labdiffs >= pos_label_diffs.view(-1, 1), 1, -1)
            pos_log_probs = pos_logits - (logits_diffs * flipped_signs).mean(dim=-1)  # 2bs
            loss += (pos_log_probs / (pos_num * (pos_num - 1))).sum()

        return loss
    

if __name__ == "__main__":
    set_seed(322)
    features = torch.rand([256, 2, 512]).float()
    labels = torch.randint(1, 80, [256, 1]).float()
    loss = PairwiseRankingLoss(eps=0.2).cuda()
    
    # times = []
    # for i in range(500):
    #     times.append(loss(features.cuda(), labels.cuda()))
    # print(sum(times) / len(times))

    print(loss.forward(features.cuda(), labels.cuda(), labels.cuda()))