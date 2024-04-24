import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from utils import get_spherical_coordinates


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
        """Compute correlation loss between input and the target
        Args:
            input (torch.Tensor): size [2B, 1]
            target (torch.Tensor): size [2B, 1]
            return_covariance (bool): True for returning covariance as the loss, otherwise will return the correlation
        Returns:
            torch.Tensor: size [label_dim]
        """
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
    def __init__(self, delta: float = 0.1, temperature: float = 2, label_diff: str = 'l1') -> None:
        super(DeltaOrderLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        assert 0 < delta < 1, f"The valid range of delta is (0,1), you assign delta as {delta}"
        self.delta = delta
    
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
        N, D = features.shape
        label_diffs = self.label_diff_fn(labels)
        feature_dists = (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)      

        # Remove diagonal
        mask = (1 - torch.eye(N).to(features.device)).bool()
        label_diffs = label_diffs.masked_select(mask).view(N, N - 1)
        feature_dists = feature_dists.masked_select(mask).view(N, N - 1)
        
        # Sorting features by their relative label differences
        with torch.no_grad():
            sorted_label_diffs, sorted_indices = torch.sort(label_diffs, dim=1)
            ranks = torch.stack(
                [ torch.unique_consecutive(lab_diff, return_inverse=True)[-1] for lab_diff in sorted_label_diffs ], 
                dim=0
            )

        feature_dists = torch.gather(feature_dists, 1, sorted_indices)
        feature_dists_diffs = feature_dists[:, None, :] - feature_dists[:, :, None] # [N, N-1, N-1]
        loss = 0.0
        for k in range(N-2):
            _lab_diffs = sorted_label_diffs[:, k, None] # [N, 1]
            _dists_diffs = feature_dists_diffs[:, k, :] # [N, N-1]

            # Compute postive logits
            pos_mask = (_lab_diffs == sorted_label_diffs) # [N, N-1]
            pos_mask[:, k] = False 
            pos_margin_mask = _dists_diffs.abs() > self.delta
            pos_logits = ((-1 * _dists_diffs.abs()).exp() * (pos_margin_mask & pos_mask)).sum(1)
            pos_logits[pos_logits==0] = 1.0
                
            # Compute negative logits
            neg_mask = (_lab_diffs < sorted_label_diffs)
            neg_margins = (ranks - ranks[:, k, None]).div(self.delta) # [B x N-1]
            neg_margin_mask = _dists_diffs > -1 * neg_margins
            neg_logits = (_dists_diffs.exp() * (neg_margin_mask & neg_mask)).sum(1)
            neg_logits[neg_logits==0] = 1.0
            loss += -1 * (pos_logits / neg_logits).log().sum()

        loss /= (N * (N-1))

        return loss


if __name__ == "__main__":
    features = torch.rand([3, 2, 512], dtype=float)
    labels = torch.randint(1, 80, [3, 1])
    print(labels)
    loss = DeltaOrderLoss().cuda()
    print(loss(features.cuda(), labels.cuda()))