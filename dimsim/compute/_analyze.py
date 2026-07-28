from __future__ import annotations

from parsl import File

from dimsim.configs.liquid import BulkLiquid


def _run_density_analysis(
    compute_config: BulkLiquid,
    production_future: dict[str, tuple[File, ...]],
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
    production_files: tuple[File, ...] = production_future["simulation_files"]  # type: ignore[assignment]

    dataframe = pandas.read_csv(production_files[2].filepath)

    estimate = dataframe["Density (g/mL)"].mean()
    logging.info(f"Estimated mean density: {estimate:.4f} g/mL")

    return {"mean_density": estimate}
