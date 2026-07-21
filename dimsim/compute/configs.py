import os

from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider, SlurmProvider
from parsl.utils import get_all_checkpoints


def local_config():
    """For development and testing."""
    if False:
        # If openff-packmol/temporary_cd was threadsafe, better to use
        # ThreadPoolExecutor and local config
        return Config(
            executors=[ThreadPoolExecutor(label="local")],
            checkpoint_mode="task_exit",
            checkpoint_files=get_all_checkpoints(),
        )

    return Config(
        executors=[
            HighThroughputExecutor(
                label="local_process_pool",
                # LocalProvider runs on the local machine
                provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                ),
                # Optional: Specify exact max workers (processes)
                max_workers_per_node=int(os.cpu_count() / 2),
            )
        ],
        strategy=None,  # Disable dynamic scaling for simpler local execution
    )


def slurm_config(partition, max_blocks=100):
    """For production HPC runs."""
    return Config(
        executors=[
            HighThroughputExecutor(
                label="slurm_gpu",
                provider=SlurmProvider(
                    partition=partition,
                    walltime="02:00:00",
                    nodes_per_block=1,
                    min_blocks=0,
                    max_blocks=max_blocks,
                    scheduler_options="#SBATCH --gres=gpu:1",
                    worker_init="source activate myenv",
                ),
            )
        ],
        checkpoint_mode="task_exit",
        checkpoint_files=get_all_checkpoints(),
    )
