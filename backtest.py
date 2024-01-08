from dataclasses import dataclass
from typing import Tuple

import polars as pl
import numpy as np
import datetime as dt


@dataclass
class BacktestArgs:
    star_date: dt.datetime
    end_date: dt.datetime
    shift: int
    lenght: dt.timedelta
    pred_lenght: int
    delta: int
    trading_days: pl.Series
    df_per_day_constituents: pl.DataFrame


class Backtester:
    def __init__(
        self,
        backtest_args: BacktestArgs,
        df_returns: pl.DataFrame,
        df_calendar: pl.DataFrame,
    ):
        # self._initialize_counters(backtest_args=backtest_args)
        self.arr_trading_days = self._compute_trading_days(
            df_calendar=df_calendar, backtest_args=backtest_args
        )
        self.df_returns = df_returns

    def _compute_trading_days(
        self, df_calendar: pl.DataFrame, backtest_args: BacktestArgs
    ):
        return (
            df_calendar.filter(
                pl.col("date").is_between(
                    backtest_args.star_date, backtest_args.end_date
                )
            )
            .select("date")
            .to_series()
            .to_numpy()
        )

    def _get_current_date_assets(
        self, current_date: dt.datetime, df_per_day_const: pl.DataFrame
    ) -> list:
        return (
            df_per_day_const.filter(pl.col("date") == current_date)
            .select("asset_name")
            .to_series()
            .to_numpy()
        )

    def _compute_date(
        self, backtest_args: BacktestArgs, actual_idx: int
    ) -> Tuple:
        return (
            backtest_args.trading_days[actual_idx],
            backtest_args.trading_days[actual_idx - backtest_args.lenght],
            backtest_args.trading_days[actual_idx - backtest_args.pred_lenght],
            backtest_args.trading_days[actual_idx + 1],
            backtest_args.trading_days[actual_idx + backtest_args.delta + 1],
        )

    def _get_current_date_returns(
        self,
        df_returns: pl.DataFrame,
        current_date: dt.datetime,
        lenght_date: dt.timedelta,
    ) -> np.array:
        return (
            df_returns.filter(
                pl.col("date").is_between(
                    current_date - lenght_date, current_date
                )
            )
            .drop("date")
            .transpose()
            .to_numpy()
        )

    def _get_pnl_date_returns(
        self,
        df_returns: pl.DataFrame,
        current_date: dt.datetime,
        lenght_date: dt.timedelta,
    ) -> np.array:
        return (
            df_returns.filter(
                pl.col("date").is_between(
                    current_date - lenght_date, current_date
                )
            )
            .drop("date")
            .transpose()
            .to_numpy()
        )

    def backtest_strategy(
        self,
        backtest_args: BacktestArgs,
        clustering_recipe: list,
        lead_lag_args: LeadLagArgs,
        df_returns: pl.DataFrame,
    ):
        pnl_out = []
        for i in range(
            backtest_args.lenght,
            len(self.arr_trading_days),
            backtest_args.shift,
        ):
            (
                current_date,
                lenght_date,
                pred_date,
                date_low_pnl,
                date_up_pnl,
            ) = self._compute_date(backtest_args=backtest_args, actual_idx=i)

            current_date_assets = self._get_current_date_assets(
                current_date=current_date,
                df_per_day_const=backtest_args.df_per_day_constituents,
            )

            (arr_returns_temp,) = self._get_current_date_returns(
                df_returns=df_returns.select(["date"] + current_date_assets),
                current_date=current_date,
                lenght=lenght_date,
            )

            cluster_label = compute_clustering(
                elem_to_cluster=arr_returns_temp,
                clustering_recipe=clustering_recipe,
            )

            lead_lag_matrix = compute_pred_lead_lag_matrix(
                arr_rets=arr_returns_temp,
                cluster_label=cluster_label,
                lag_method=lead_lag_args.lag_method,
                window=lead_lag_args.lag_window,
            )

            df_lead_lag_group = compute_asset_group(
                lead_lag_matrix=lead_lag_matrix,
                current_date_asset=current_date_assets,
                cluster_label=cluster_label,
                leaders_threshold=lead_lag_args.leader_threshold,
                laggers_threshold=lead_lag_args.laggers_threshold,
            )

            df_pred = compute_prediction(
                df_lead_lag_group=df_lead_lag_group, df_rets=df_returns
            )

        return None
