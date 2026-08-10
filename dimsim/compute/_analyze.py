from __future__ import annotations

from dimsim.compute._files import ProductionFiles


def _run_density_analysis(
    production_future: dict[str, ProductionFiles],
    job_dir: str,
) -> dict[str, float]:

    import pandas

    from dimsim.compute._logging import _set_up_logger

    logger = _set_up_logger(f"{job_dir}/analyze.log")

    logger.info("Starting density analysis")

    # TODO: Double check for equilibration/stability
    production_files: ProductionFiles = production_future["simulation_files"]

    dataframe = pandas.read_csv(production_files["state_data"].filepath)

    mean = dataframe["Density (g/mL)"].mean()
    std = dataframe["Density (g/mL)"].std()

    logger.info(f"Estimated mean density: {mean:.4f} +/- {std:.4f} g/mL")

    return {
        "mean": mean,
        "std": std,
    }
