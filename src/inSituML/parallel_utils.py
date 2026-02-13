import os, sys


def get_world_size():
    """Get the world size for distributed training."""
    world_size = None
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_NTASKS" in os.environ:
        print(
            (
                "[WW] WORLD_SIZE not defined in env, "
                + "falling back to SLURM_NTASKS."
            ),
            file=sys.stderr,
        )
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        raise RuntimeError("cannot determine WORLD_SIZE")
    return world_size

