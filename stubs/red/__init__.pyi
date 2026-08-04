from collections.abc import Callable
from pathlib import Path

import numpy as _np
import numpy.typing as _npt

def detect_equilibration_window(
    data: _npt.NDArray[_np.float64],
    times: _npt.NDArray[_np.float64] | None = None,
    method: str = "min_sse",
    kernel: Callable[[int], _npt.NDArray[_np.float64]] = _np.bartlett,
    window_size_fn: Callable[[int], int] | None = lambda x: round(x**0.5),
    window_size: int | None = None,
    frac_padding: float = 0.1,
    plot: bool = False,
    plot_name: str | Path = "equilibration_sse_window.png",
    time_units: str = "ns",
    data_y_label: str = r"$\Delta G$ / kcal mol$^{-1}$",
    plot_window_size: bool = True,
) -> tuple[float | int, float, float]: ...
