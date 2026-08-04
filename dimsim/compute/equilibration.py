import numpy
import polars
import red


def assert_equilibrated(simulation_data: polars.DataFrame) -> bool:
    idx, _g, ess = red.detect_equilibration_window(
        numpy.array(simulation_data["Potential Energy (kJ/mole)"]),
        method="min_sse",
        plot=True,
    )

    return (ess > 50) and (idx < len(simulation_data) * 0.5)
