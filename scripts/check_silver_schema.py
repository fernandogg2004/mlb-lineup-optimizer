"""Verificación rápida del schema Silver tras un rebuild completo."""
import polars as pl

NEEDED = ["game_pk", "at_bat_number", "home_team", "is_home", "pitch_count_in_pa"]

total = 0
for y in range(2015, 2027):
    df = pl.read_parquet(f"data/silver/plate_appearances/season={y}/data.parquet")
    ok = all(c in df.columns for c in NEEDED)
    total += len(df)
    nulls = df["is_home"].null_count() if "is_home" in df.columns else "N/A"
    print(f"{y}  {len(df):>7,} PAs  contexto: {ok}  is_home nulls: {nulls}  "
          f"max: {str(df['game_date'].max())[:10]}")
print(f"TOTAL: {total:,} PAs")
