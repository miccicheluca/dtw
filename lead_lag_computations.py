from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import polars as pl
from dtaidistance import dtw

from ll_computation import compute_ll_from_arr_tuple, gen_lead_lag_matrix


@dataclass
class LeadLagArgs:
    lag_method: str
    lag_window: int
    leader_threshold: float
    laggers_threshold: float


def _compute_path(
    arr_rets_1: np.array, arr_rets_2: np.array, args: Dict
) -> List:
    return dtw.warping_path_fast(from_s=arr_rets_1, to_s=arr_rets_2, **args)


def compute_pred_lead_lag_matrix(
    cluster_label: np.array,
    arr_rets: np.array,
    lag_method: str,
    window: int,
) -> np.matrix:
    n_asset = len(cluster_label)
    same_cluster_paths = np.array(
        [
            _compute_path(
                arr_rets_1=arr_rets[i],
                arr_rets_2=arr_rets[j],
                args={"window": window + 1},
            )
            for i in range(n_asset)
            for j in range(i + 1, n_asset)
            if cluster_label[i] == cluster_label[j]
        ],
        dtype=np.double,
    )

    median_from_paths = compute_ll_from_arr_tuple(
        arr=same_cluster_paths, lag_method=lag_method
    )

    return gen_lead_lag_matrix(
        clusters=cluster_label,
        n_asset=n_asset,
        median_from_paths=median_from_paths,
    )
