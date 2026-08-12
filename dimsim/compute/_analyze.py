from __future__ import annotations


def _run_density_analysis(
    job_dir: str,
) -> dict[str, float]:
    """Run a naive density analysis from recorded simulation data."""
    import pandas

    from dimsim.compute._logging import _set_up_logger

    logger = _set_up_logger(f"{job_dir}/analyze.log")

    logger.info("Starting density analysis")

    # TODO: Double check for equilibration/stability
    try:
        dataframe = pandas.read_csv(f"{job_dir}/production.csv")
    except TypeError:
        logger.error(f"Production run CSV data file not found in future, we are in {job_dir=}")
        raise

    mean = dataframe["Density (g/mL)"].mean()
    std = dataframe["Density (g/mL)"].std()

    logger.info(f"Estimated mean density: {mean:.4f} +/- {std:.4f} g/mL")

    return {
        "mean": mean,
        "std": std,
    }
