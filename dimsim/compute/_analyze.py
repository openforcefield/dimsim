from __future__ import annotations

from dimsim.compute._files import ProductionFiles
from dimsim.configs.liquid import BulkLiquid


def _run_density_analysis(
    compute_config: BulkLiquid,
    production_future: dict[str, ProductionFiles],
    job_dir: str,
) -> dict[str, float]:
    import logging

    import pandas

    logging.basicConfig(
        filename=f"{job_dir}/simulation.log",
        level=logging.INFO,
    )
    logging.info("Starting density analysis")

    # TODO: Double check for equilibration/stability
    production_files: ProductionFiles = production_future["simulation_files"]

    dataframe = pandas.read_csv(production_files["log"].filepath)

    estimate = dataframe["Density (g/mL)"].mean()
    logging.info(f"Estimated mean density: {estimate:.4f} g/mL")

    return {"mean_density": estimate}
