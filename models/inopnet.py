import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Union
#sys.path.append('/opt/data/private/dzt/project/withmodelnet40/src')
from common.torch import to_numpy
from models.pointnet_util import square_distance, angle_difference
from models.feature_nets import FeatExtractionEarlyFusion
from models.feature_nets import ParameterPredictionNet
from common.math_torch import se3


_EPS = 1e-5  # To prevent division by zero


def match_features(feat_src, feat_ref, metric='l2'):
    """ Compute pairwise distance between features

    Args:
        feat_src: (B, J, C)
        feat_ref: (B, K, C)
        metric: either 'angle' or 'l2' (squared euclidean)

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
         in the src agrees with every point in the ref.
    """
    assert feat_src.shape[-1] == feat_ref.shape[-1]

    if metric == 'l2':
        dist_matrix = square_distance(feat_src, feat_ref)
    elif metric == 'angle':
        feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + _EPS)
        feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + _EPS)

        dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
    else:
        raise NotImplementedError

    return dist_matrix


def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1, ignore_t:Union[float,torch.Tensor] = None) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)
        if ignore_t is not None:
            _,N,M = log_alpha_padded.shape
            ignore_t = ignore_t[:,None,None].repeat(1,N,M)
            log_alpha_padded[:,:,-1] = ignore_t[:,:,-1]
            log_alpha_padded[:,-1,:] = ignore_t[:,-1,:]

            
        

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):  # SVD
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_a = torch.sum(a * weights_normalized, dim=1)  # (B,3)
    centroid_b = torch.sum(b * weights_normalized, dim=1)  # (B,3)
    a_centered = a - centroid_a[:, None, :]  # centered (B,M,3)
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)  # co-var (B,3,3)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform


class INOPNet(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.add_slack = not args.no_slack
        self.num_sk_iter = args.num_sk_iter
        self.weights_net = ParameterPredictionNet()  # get anneling parameter
        self.feat_extractor = FeatExtractionEarlyFusion(
            features=args.features, feature_dim=args.feat_dim,
            radius=args.radius, num_neighbors=args.num_neighbors)  # feature extract


    def compute_affinity(self, beta, feat_distance, alpha=0.5):
        """Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
        if isinstance(alpha, float):
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
        else:
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
        return hybrid_affinity

    def forward(self, data, num_iter: int = 1):
        """Forward pass for RPMNet

        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, 6)
                    'points_ref': Reference points (B, K, 6)
            num_iter (int): Number of iterations. Recommended to be 2 for training

        Returns:
            transform: Transform to apply to source points such that they align to reference
            src_transformed: Transformed source points
        """
        endpoints = {}

        xyz_ref, norm_ref = data['points_ref'][:, :, :3], data['points_ref'][:, :, 3:6]
        xyz_src, norm_src = data['points_src'][:, :, :3], data['points_src'][:, :, 3:6]
        xyz_src_t, norm_src_t = xyz_src, norm_src

        transforms = []
        all_gamma, all_perm_matrices, all_weighted_ref = [], [], []
        all_beta, all_alpha = [], []
        for i in range(num_iter):

            beta, alpha, ignore_t = self.weights_net([xyz_src_t, xyz_ref])
            #ignore_t = torch.clamp(ignore_t,max=beta*alpha-beta)
            feat_src = self.feat_extractor(xyz_src_t, norm_src_t)
            feat_ref = self.feat_extractor(xyz_ref, norm_ref)

            feat_distance = match_features(feat_src, feat_ref)  # feature dist
            affinity = self.compute_affinity(beta, feat_distance, alpha=alpha)  # annelling dist

            # Compute weighted coordinates
            log_perm_matrix = sinkhorn(affinity, n_iters=self.num_sk_iter, slack=self.add_slack, ignore_t=ignore_t)  # sinkhorn norm
            perm_matrix = torch.exp(log_perm_matrix)

            B,N,M = perm_matrix.shape
            max_idx = torch.argmax(perm_matrix,dim=-1)
            one2one_mask = max_idx != M  # B N

            batch_transform = []
            batch_all_perm_matrices = []
            batch_all_weighted_ref = []
            for b in range(B):
                xyz_src_b = xyz_src[b:b+1][one2one_mask[b:b+1]].unsqueeze(0)
                perm_matrix_b = perm_matrix[b:b+1][one2one_mask[b:b+1]].unsqueeze(0)[...,:-1]
                weighted_ref_b = perm_matrix_b @ xyz_ref[b:b+1] / (torch.sum(perm_matrix_b, dim=2, keepdim=True) + _EPS)
                transform_b = compute_rigid_transform(xyz_src_b, weighted_ref_b, weights=torch.sum(perm_matrix_b, dim=2))
                batch_transform.append(transform_b)
                batch_all_perm_matrices.append(perm_matrix_b)
                batch_all_weighted_ref.append(weighted_ref_b)



            #weighted_ref = perm_matrix[:,:,:-1] @ xyz_ref / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)
            # Compute transform and transform points
            #transform = compute_rigid_transform(xyz_src_1, weighted_ref, weights=torch.sum(perm_matrix, dim=2))
            transform = torch.cat(batch_transform,dim=0)
            xyz_src_t, norm_src_t = se3.transform(transform.detach(), xyz_src, norm_src)

            transforms.append(transform)
            all_gamma.append(torch.exp(affinity))
            all_perm_matrices.append(batch_all_perm_matrices)
            all_weighted_ref.append(batch_all_weighted_ref)
            all_beta.append(to_numpy(beta))
            all_alpha.append(to_numpy(alpha))

        endpoints['perm_matrices_init'] = all_gamma
        endpoints['perm_matrices'] = all_perm_matrices
        endpoints['weighted_ref'] = all_weighted_ref
        endpoints['beta'] = np.stack(all_beta, axis=0)
        endpoints['alpha'] = np.stack(all_alpha, axis=0)

        return transforms, endpoints


def get_model(args: argparse.Namespace) -> INOPNet:
    return INOPNet(args)


if __name__ == "__main__":
    import torch
    a = torch.randn(2,5,5)
    res = sinkhorn(a,5,True)
    idx = torch.argmax(res,dim=2)
    mask = idx == res.shape[2]-1
    print(mask)