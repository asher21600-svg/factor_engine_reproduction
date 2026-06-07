"""Seed factor — paper Listing 1.3, verbatim (report-bootstrapped seed program).

A price-volume "pressure" factor: three intraday money-flow style signals
(sf1: volume x close-position-in-range; sf2: volume x upper-shadow;
sf3: volume x lower-body), each cross-sectionally z-scored per day and combined
with weights (w1,w2,w3).  This is the seed of FE's initial pool.
"""

# The canonical source string IS the evolution genome.  Kept faithful to the
# paper; only cosmetic indentation is normalized.
SEED_SRC = r'''
def factor(pricing_data, parameters):
    w1 = parameters.get("w1", 0.25)
    w2 = parameters.get("w2", 0.25)
    w3 = parameters.get("w3", 0.50)
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
    ])

    daily_range_expr = pl.col('high') - pl.col('low')
    sf1_expr = -pl.col('volume') * (pl.col('close') - pl.col('low')) / (daily_range_expr + EPSILON)
    sf2_expr = -pl.col('volume') * (pl.col('high') - pl.col('open')) / (daily_range_expr + EPSILON)
    sf3_expr = pl.col('volume') * (pl.min_horizontal('open', 'close') - pl.col('low')) / (daily_range_expr + EPSILON)

    df_factor = df_pl.with_columns(
        z1=(sf1_expr - sf1_expr.mean().over('datetime')) / (sf1_expr.std(ddof=0).over('datetime') + EPSILON),
        z2=(sf2_expr - sf2_expr.mean().over('datetime')) / (sf2_expr.std(ddof=0).over('datetime') + EPSILON),
        z3=(sf3_expr - sf3_expr.mean().over('datetime')) / (sf3_expr.std(ddof=0).over('datetime') + EPSILON),
    ).with_columns(
        (w1 * pl.col('z1') + w2 * pl.col('z2') + w3 * pl.col('z3')).alias('Factor')
    )

    df_tf = df_factor.select(['instrument', 'datetime', 'Factor'])
    df_tf = df_tf.filter(pl.col('Factor').is_not_nan() & pl.col('Factor').is_finite())
    df_tf = df_tf.with_columns(pl.col("datetime").cast(pl.Date).alias("datetime"))
    return df_tf
'''


def seed_factor(pricing_data, parameters):
    """Importable handle (compiles SEED_SRC lazily)."""
    from .contract import compile_factor
    return compile_factor(SEED_SRC)(pricing_data, parameters)
