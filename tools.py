from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy import random as rd
from dtaidistance import dtw
from scipy.spatial.distance import cdist
from kmedoids import KMedoids
from sklearn.cluster import KMeans


@dataclass
class DistArgs:
    method: str
    args: dict


@dataclass
class ClusteringArgs:
    cluster_func: "function"
    clustering_args: dict


def compute_true_lead_lag_matrix(
    cluster_label: np.array, arr_lag: np.array
) -> np.matrix:
    n_asset = len(cluster_label)
    lead_lag_matrix = np.zeros((n_asset, n_asset))
    for i in range(n_asset):
        for j in range(i + 1, n_asset):
            if cluster_label[i] == cluster_label[j]:
                lead_lag_value = arr_lag[i] - arr_lag[j]
                lead_lag_matrix[i][j] = -lead_lag_value
                lead_lag_matrix[j][i] = lead_lag_value
    return lead_lag_matrix


def generate_k_factor_data(
    n_points: int, std: float, max_lag: int, n_assets: int, k_factor: int
) -> Dict:
    """Generate factor model returns"""

    factors = rd.normal(0, 1, size=(k_factor, n_points))
    lags_label_threshold = (n_assets / (max_lag + 1)) / k_factor
    white_noise_per_asset = rd.normal(0, std, size=(n_assets, n_points))
    factors_belong = np.array(
        [int((i - 1) / (n_assets / k_factor)) for i in range(1, n_assets + 1)]
    )
    lags_per_asset_per_factor = [
        int((i - 1) / lags_label_threshold)
        for i in range(1, int(n_assets / k_factor) + 1)
    ]
    lags_per_asset = np.concatenate(
        [lags_per_asset_per_factor for _ in range(k_factor)]
    )

    asset_rets = np.array(
        [
            np.roll(a=factors[factor], shift=lag) + white_noise
            for factor, lag, white_noise in zip(
                factors_belong, lags_per_asset, white_noise_per_asset
            )
        ]
    )
    return {
        "lag": lags_per_asset,
        "cluster": factors_belong,
        "rets": asset_rets,
        "lead_lag_matrix": compute_true_lead_lag_matrix(
            cluster_label=factors_belong, arr_lag=lags_per_asset
        ),
    }


def compute_dist_matrix(arr_rets: np.array, dist_args: DistArgs) -> np.matrix:
    if dist_args.method in ["dtw"]:
        dist_matrix = dtw.distance_matrix_fast(s=arr_rets, **dist_args.args)
    elif dist_args.method in ["euclidean", "cityblock", "cosine"]:
        dist_matrix = cdist(XA=arr_rets, XB=arr_rets, metric=dist_args.method)
    else:
        dist_matrix = arr_rets
    return dist_matrix


def standardize_clusters(clusters: List[List[int]], n_elem: int) -> List[int]:
    i = 0
    res = np.zeros(n_elem)
    for l in clusters:
        np.put(a=res, ind=l, v=i)
        i += 1
    return res


def compute_clustering(
    elem_to_cluster: np.array, clustering_recipe: List
) -> List:
    clust_obj = clustering_recipe[0](
        elem_to_cluster, clustering_recipe[1], tolerance=0.001
    )
    clust_obj.process()

    return standardize_clusters(
        clusters=clust_obj.get_clusters(), n_elem=len(elem_to_cluster)
    )


def compute_clusters_from_dist_matrix(
    arr_rets: np.array, dist_args: DistArgs, cluster_recipe: List
) -> pl.DataFrame:
    """Compute cluster from distance matrix"""
    # 1) compute distance matrix
    arr_dist = compute_dist_matrix(arr_rets=arr_rets, dist_args=dist_args)
    # dist_arr = dtw.distance_matrix_fast(arr_rets)
    # 2) compute
    clusters = compute_clustering(
        elem_to_cluster=arr_dist, clustering_recipe=cluster_recipe
    )
    return clusters


def compute_lead_lag_matrix():
    # # 3) compute Lead/Lag Matrix per cluster
    # n_assets = len(arr_rets)
    # lead_lag_matrix = np.zeros(shape=(n_assets, n_assets))
    # for i in range(n_assets):
    #     for j in range(i, n_assets):
    #         lead_lag_matrix[i,j], lead_lag_matrix[j,i] = compute_lead_lag_per_asset()
    return None


def generate_true_lag_data(
    n_points: int, std: float, max_lag: int, n_assets: int, k_factor: int
) -> dict:
    """Generate factor model returns"""

    factors = rd.normal(0, 1, size=(k_factor, n_points))
    lags_label_threshold = n_assets // 2
    white_noise_per_asset = rd.normal(0, std, size=(n_assets, n_points))
    factors_belong = np.array(
        [int((i - 1) / (n_assets / k_factor)) for i in range(1, n_assets + 1)]
    )
    lags_per_asset_per_factor = [
        int((i - 1) // (lags_label_threshold / k_factor))
        for i in range(1, int(n_assets / k_factor) + 1)
    ]
    lags_per_asset = (
        np.concatenate([lags_per_asset_per_factor for _ in range(k_factor)])
        * max_lag
    )

    asset_rets = np.array(
        [
            np.roll(a=factors[factor], shift=lag) + white_noise
            for factor, lag, white_noise in zip(
                factors_belong, lags_per_asset, white_noise_per_asset
            )
        ]
    )
    return {
        "lag": lags_per_asset,
        "cluster": factors_belong,
        "rets": asset_rets,
        "lead_lag_matrix": compute_true_lead_lag_matrix(
            cluster_label=factors_belong, arr_lag=lags_per_asset
        ),
    }
