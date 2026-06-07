"""Evolved factor — paper Listing 1.4, after 40 evolution iterations.

Refinements over the seed (all stated in the paper's caption):
  * turnover = volume * close   (capital-weighted instead of raw volume)
  * close vs MID-price  ((high+low)/2)  instead of close vs low
  * rank-normalization of each component (robust to volume outliers)
  * EWM temporal smoothing per instrument (span=smoothing_window)
  * final cross-sectional z-score

BUG FIX vs the printed listing: Listing 1.4 uses `daily_range_expr` but never
defines it (the definition lived in the seed listing).  We restore the obvious
definition `high - low`, matching the seed.  This is annotated below.
"""

EVOLVED_SRC = r'''
def trend_factor(pricing_data, parameters):
    w3 = parameters.get("w3", 0.50)
    w1 = parameters.get("w1", (1.0 - w3) / 2.0)
    w2 = parameters.get("w2", (1.0 - w3) / 2.0)
    smoothing_window = parameters.get("smoothing_window", 5)
    EPSILON = parameters.get("epsilon", 1e-9)

    if isinstance(pricing_data, pd.DataFrame):
        df_pl = pl.from_pandas(pricing_data.reset_index()).rename({
            '$close': 'close', '$open': 'open', '$high': 'high',
            '$low': 'low', '$volume': 'volume'})
    else:
        df_pl = pricing_data.rename({
            '$close': 'close', '$open': 'open', '$high': 'high',
            '$low': 'low', '$volume': 'volume'})

    df_pl = df_pl.select(
        ['instrument', 'datetime', 'open', 'high', 'low', 'close', 'volume']
    ).with_columns([
        pl.col("datetime").cast(pl.Date),
        pl.col(['open', 'high', 'low', 'close', 'volume']).cast(pl.Float64),
    ]).sort(['instrument', 'datetime'])

    daily_range_expr = pl.col('high') - pl.col('low')  # BUG FIX: missing in Listing 1.4
    turnover_expr = pl.col('volume') * pl.col('close')  # capital-weighted signal
    sf1_expr = -turnover_expr * (pl.col('close') - (pl.col('high') + pl.col('low')) / 2.0) / (daily_range_expr + EPSILON)
    sf2_expr = -turnover_expr * (pl.col('high') - pl.col('open')) / (daily_range_expr + EPSILON)
    sf3_expr = turnover_expr * (pl.min_horizontal('open', 'close') - pl.col('low')) / (daily_range_expr + EPSILON)

    rank_norm_expr = lambda expr: (expr.rank(method='average').over('datetime') / (expr.count().over('datetime') + 1)) - 0.5

    df_factor = df_pl.with_columns(
        raw_combined_factor=(
            w1 * rank_norm_expr(sf1_expr) +
            w2 * rank_norm_expr(sf2_expr) +
            w3 * rank_norm_expr(sf3_expr))
    ).with_columns(
        smoothed_factor=pl.col('raw_combined_factor').ewm_mean(
            span=smoothing_window, min_periods=max(1, smoothing_window // 2)).over('instrument')
    ).with_columns(
        Factor=(
            (pl.col('smoothed_factor') - pl.col('smoothed_factor').mean().over('datetime')) /
            (pl.col('smoothed_factor').std(ddof=0).over('datetime') + EPSILON))
    )

    df_tf = df_factor.select(['instrument', 'datetime', 'Factor'])
    df_tf = df_tf.filter(pl.col('Factor').is_not_nan() & pl.col('Factor').is_finite())
    df_tf = df_tf.with_columns(pl.col("datetime").cast(pl.Date).alias("datetime"))
    return df_tf
'''


def evolved_factor(pricing_data, parameters):
    """Importable handle (compiles EVOLVED_SRC lazily)."""
    from .contract import compile_factor
    return compile_factor(EVOLVED_SRC)(pricing_data, parameters)
