import polars as pl
import numpy as np


def _create_df_lead_lag(
    lead_lag_matrix: np.array,
    current_date_asset: np.array,
    cluster_label: np.array,
) -> pl.DataFrame:
    row_sum_ll = pl.DataFrame(lead_lag_matrix).sum_horizontal()
    return pl.DataFrame(
        {
            "assets": current_date_asset,
            "cluster_label": cluster_label,
            "row_sum": row_sum_ll,
        }
    )


def compute_asset_group(
    lead_lag_matrix: np.array,
    current_date_asset: np.array,
    cluster_label: np.array,
    leaders_threshold: float,
    laggers_threshold: float,
) -> pl.DataFrame:
    """Leaders are indentified as 1, laggers as -1, others as 0"""
    df_lead_lag = _create_df_lead_lag(
        lead_lag_matrix=lead_lag_matrix,
        current_date_asset=current_date_asset,
        cluster_label=cluster_label,
    )

    return df_lead_lag.with_columns(
        pl.when(
            pl.col("row_sum").rank(method="dense").over("cluster_label")
            < pl.col("row_sum").rank(method="dense").max().over("cluster_label")
            * leaders_threshold
        )
        .then(1)
        .when(
            pl.col("row_sum").rank(method="dense").over("cluster_label")
            > pl.col("row_sum").rank(method="dense").max().over("cluster_label")
            * laggers_threshold
        )
        .then(-1)
        .otherwise(0)
        .alias("group")
    ).filter(pl.col("group").is_in([1, -1]))
