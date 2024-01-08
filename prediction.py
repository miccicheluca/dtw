import polars as pl
import numpy as np


def compute_ewma(df_rets: pl.DataFrame, alpha: float):
    return df_rets.mean_horizontal().ewm_mean(alpha=alpha)


def compute_pred_per_cluster(
    df_lead_lag_group: pl.DataFrame, df_rets: pl.DataFrame, alpha: float
):
    clusters = df_lead_lag_group.select("group").to_series().unique()
    out = []
    for cluster in clusters:
        group_leaders_assets = (
            df_lead_lag_group.filter(pl.col("cluster_label") == cluster)
            .select("asset_name")
            .to_series()
            .to_numpy()
        )
        out.append(
            pl.DataFrame(
                {
                    "cluster_label": cluster,
                    "pred": compute_ewma(
                        df_rets=df_rets.select(group_leaders_assets),
                        alpha=alpha,
                    ),
                }
            )
        )
    return pl.concat(out)


def compute_prediction(
    df_lead_lag_group: pl.DataFrame,
    df_rets: pl.DataFrame,
    per_cluster: bool,
    alpha: float,
):
    leaders_assets = (
        df_lead_lag_group.filter(pl.col("group") == 1)
        .select("asset_name")
        .to_series()
        .to_numpy()
    )

    if per_cluster:
        df_pred = compute_pred_per_cluster(
            df_lead_lag_group=df_lead_lag_group, df_rets=df_rets, alpha=alpha
        )
    else:
        df_pred = pl.DataFrame(
            {
                "pred": compute_ewma(
                    df_rets=df_rets.select(leaders_assets), alpha=alpha
                )
            }
        )
    return df_pred
