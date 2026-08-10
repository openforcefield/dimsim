from __future__ import annotations

from dimsim.compute._files import ProductionFiles


def _run_density_analysis(
    production_future: dict[str, ProductionFiles],
    job_dir: str,
) -> dict[str, float]:
    import logging

    import pandas

    logger = logging.getLogger("dimsim")  # same package name
    logger.handlers.clear()  # worker starts fresh, but be safe
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{job_dir}/produce.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

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
