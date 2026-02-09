"""
General utilities for testing
"""

import pathlib


def get_test_data_path(path: str | pathlib.Path) -> pathlib.Path:
    """Get the path to a test data file or directory

    Parameters
    ----------
    path : str | pathlib.Path
        The relative path to the test data file,
        relative to the `dimsim/_tests/data` directory.

    Returns
    -------
    pathlib.Path
        The absolute path to the test data file.
    """
    return pathlib.Path(__file__).parent / "data" / path
