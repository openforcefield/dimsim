import pathlib


class TrajectoryStore:
    """
    Class to manage storage paths for simulation trajectories.

    Parameters
    ----------
    base_path : pathlib.Path
        The base directory where trajectories will be stored.
    file_pattern : str
        The pattern for naming trajectory files, which can include placeholders
        such as {{iteration}}, {{replicate}}, and {{uuid}}.
    """

    def __init__(
        self,
        base_path: pathlib.Path = "trajectories",
        file_pattern: str = "{iteration}/{replicate}/{uuid}.msgpack"
    ):
        self.base_path = base_path
        self.file_pattern = file_pattern
