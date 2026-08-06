import pyarrow.parquet

from dimsim.configs.liquid import BulkLiquid, make_liquid_table

entries: list[BulkLiquid] = list()
table = make_liquid_table(entries)

pyarrow.parquet.write_table(table, "compute_jobs.parquet")
