import polars as pl
import datetime as dt


class DataProcessing:
    def __init__(
        self,
        df_raw: pl.DataFrame,
        df_calendar: pl.DataFrame,
        df_constituents: pl.DataFrame,
    ):
        self.df_prep_data = self._prepare_data(
            df_raw=df_raw,
            df_calendar=df_calendar,
            df_constituents=df_constituents,
        )

    def get_prep_data(self):
        """Getters prepared data"""
        return self.df_prep_data

    def _setup_calendar(
        self, df_raw: pl.DataFrame, df_calendar: pl.DataFrame
    ) -> pl.DataFrame:
        return df_calendar.join(df_raw, on="date", how="left")

    def _prepare_data(
        self,
        df_raw: pl.DataFrame,
        df_calendar: pl.DataFrame,
        df_constituents: pl.DataFrame,
    ) -> pl.DataFrame:
        df_prep = df_calendar.filter(
            pl.col("date") > dt.datetime(2000, 1, 1)
        ).join(df_raw, on="date", how="left")

        func_recipe = [
            (self._is_constituent, {"df_constituents": df_constituents}),
            (self._compute_returns, {}),
            (self._compute_idx_returns, {}),
            (self._standardize_data, {}),
        ]

        for func, args in func_recipe:
            df_prep = df_prep.pipe(func, **args)

        return df_prep

    def _compute_returns(self, df_raw: pl.DataFrame) -> pl.DataFrame:
        return df_raw.with_columns(
            pl.col("px_last")
            .forward_fill()
            .pct_change(1)
            .over("asset_name")
            .alias("f_returns")
        )

    def _standardize_data(self, df_raw: pl.DataFrame) -> pl.DataFrame:
        return df_raw.with_columns(
            pl.when(pl.col("px_last") is None)
            .then(None)
            .otherwise(0.0)
            .alias("none_mask")
        ).with_columns(
            (
                pl.col("f_returns")
                - pl.col("f_idx_returns")
                + pl.col("none_mask")
            )
            .clip(lower_bound=-0.15, upper_bound=0.15)
            .alias("f_excess_return")
        )

    def _compute_idx_returns(self, df_raw: pl.DataFrame) -> pl.DataFrame:
        df_idx_rets = (
            df_raw.select(["date", "px_last_idx"])
            .unique("date")
            .sort("date")
            .with_columns(
                pl.col("px_last_idx")
                .forward_fill()
                .pct_change(1)
                .alias("f_idx_returns")
            )
            .select(["date", "f_idx_returns"])
        )
        return df_raw.join(df_idx_rets, on="date", how="left")

    def _is_constituent(
        self,
        df_raw: pl.DataFrame,
        df_constituents: pl.DataFrame,
    ) -> pl.DataFrame:
        """stock is index constituents point in time"""

        # add year and month
        df_temp = df_raw.with_columns(
            [
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month"),
            ]
        )
        df_constituents = df_constituents.with_columns(
            [
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month"),
                pl.lit(True).alias("in_universe"),
            ]
        )

        return (
            df_temp.join(
                df_constituents.select(
                    ["year", "month", "asset_name", "in_universe"]
                ),
                on=["asset_name", "year", "month"],
                how="left",
            )
            .with_columns(pl.col("in_universe").fill_null(False))
            .drop(["year", "month"])
        )
