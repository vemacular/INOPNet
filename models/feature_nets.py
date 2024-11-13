"""Feature Extraction and Parameter Prediction networks
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from models.pointnet_util import sample_and_group_multi

_raw_features_sizes = {'xyz': 3, 'dxyz': 3, 'ppf': 4}
_raw_features_order = {'xyz': 0, 'dxyz': 1, 'ppf': 2}

# pointnet for annelling parameters and ignore thred
class ParameterPredictionNet(nn.Module):
    def __init__(self):
        """PointNet based Parameter prediction network
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

        #self.weights_dim = weights_dim

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            #nn.Linear(256, 3 + np.prod(weights_dim)),
        )  # get anneling parameter
        self.annelling_linear = nn.Linear(256,2)
        self.feature_trans = nn.Linear(256,10)
        self.ignore_linear = nn.Linear(12,1)


    def forward(self, x):
        """ Returns alpha, beta, and gating_weights (if needed)

        Args:
            x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

        Returns:
            beta, alpha, weightings
        """

        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        postfeat = self.postpool(pooled)
        raw_weights = self.annelling_linear(postfeat)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])
        #ignore_t = F.softplus(raw_weights[:,2])
        trans_feat = self.feature_trans(postfeat)
        ignore_t = self.ignore_linear(torch.cat([trans_feat, beta.unsqueeze(1), alpha.unsqueeze(1)],dim=1))
        ignore_t = F.softplus(ignore_t[:,0])


        return beta, alpha, ignore_t


class ParameterPredictionNetConstant(nn.Module):
    def __init__(self, weights_dim):
        """Parameter Prediction Network with single alpha/beta as parameter.

        See: Ablation study (Table 4) in paper
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

        self.anneal_weights = nn.Parameter(torch.zeros(2 + np.prod(weights_dim)))
        self.weights_dim = weights_dim

        self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x):
        """Returns beta, gating_weights"""

        batch_size = x[0].shape[0]
        raw_weights = self.anneal_weights
        beta = F.softplus(raw_weights[0].expand(batch_size))
        alpha = F.softplus(raw_weights[1].expand(batch_size))

        return beta, alpha


def get_prepool(in_dim, out_dim):
    """Shared FC part in PointNet before max pooling"""
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
    )
    return net  # c:10 -> 256


def get_postpool(in_dim, out_dim):
    """Linear layers in PointNet after max pooling

    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels. Typically smaller than in_dim

    """
    net = nn.Sequential(
        nn.Conv1d(in_dim, in_dim, 1),
        nn.GroupNorm(8, in_dim),
        nn.ReLU(),
        nn.Conv1d(in_dim, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
        nn.Conv1d(out_dim, out_dim, 1),
    )

    return net

# aggregation weight net
class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [4, 4]):  # 3,8,[4,4]
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.GroupNorm(4, out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.GroupNorm(4, hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.GroupNorm(4, hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.GroupNorm(4, out_channel))
        
    def forward(self, localized_xyz):
        #xyz : B x 3 x K x N

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights =  F.relu(bn(conv(weights)))

        return weights


class FeatExtractionEarlyFusion(nn.Module):
    """3DPC: Feature extraction Module that extracts hybrid features"""
    def __init__(self, features, feature_dim, radius, num_neighbors):  # feature_dim = 128
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info('Using early fusion, feature dim = {}'.format(feature_dim))
        self.radius = radius
        self.n_sample = num_neighbors

        self.features = sorted(features, key=lambda f: _raw_features_order[f])
        self._logger.info('Feature extraction using features {}'.format(', '.join(self.features)))

        # Layers
        raw_dim = np.sum([_raw_features_sizes[f] for f in self.features])  # number of channels after concat
        self.prepool = get_prepool(raw_dim, feature_dim * 2)  # 10 -> 256  feature extract
        self.weightnet = WeightNet(3, 8)  # aggregation wight
        self.linear = nn.Linear(8 * feature_dim * 2, feature_dim * 2)
        self.bn_linear = nn.GroupNorm(8, feature_dim * 2)
        self.postpool = get_postpool(feature_dim * 2, feature_dim)

    def forward(self, xyz, normals):
        """Forward pass of the feature extraction network

        Args:
            xyz: (B, N, 3)
            normals: (B, N, 3)

        Returns:
            cluster features (B, N, C)

        """
        B, N, C = xyz.shape

        features = sample_and_group_multi(-1, self.radius, self.n_sample, xyz, normals)  # [xyz, dxyz, ppf]
        features['xyz'] = features['xyz'][:, :, None, :]

        # Gate and concat
        concat = []
        for i in range(len(self.features)):
            f = self.features[i]
            expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        fused_input_feat = torch.cat(concat, -1)  # b,N,K,10

        # Prepool_FC, pool, postpool-FC
        new_feat = fused_input_feat.permute(0, 3, 2, 1)  # [B, 10, k, N]
        new_feat = self.prepool(new_feat)

        #pooled_feat = torch.max(new_feat, 2)[0]  # Max pooling (B, C, N)
        grouped_xyz = features['dxyz'].permute(0, 3, 2, 1)  # B,3,K,N
        weights = self.weightnet(grouped_xyz)
        pooled_feat = torch.matmul(input=new_feat.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, -1)  # B N 2048
        pooled_feat = self.linear(pooled_feat)  # 256
        pooled_feat = self.bn_linear(pooled_feat.permute(0, 2, 1))  # b 256 n
        pooled_feat = F.relu(pooled_feat)

        post_feat = self.postpool(pooled_feat)  # Post pooling dense layers
        cluster_feat = post_feat.permute(0, 2, 1)
        cluster_feat = cluster_feat / torch.norm(cluster_feat, dim=-1, keepdim=True)

        return cluster_feat  # (B, N, C)


