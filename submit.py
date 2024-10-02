"""Submits a Weights & Biases sweep to Slurm."""

import argparse
import re
import subprocess
from pathlib import Path
from shutil import which

parser = argparse.ArgumentParser(description="Submit Weights & Biases sweeps to Slurm")
parser.add_argument("sweep-id", type=str, help="The Weights & Biases sweep ID")
parser.add_argument(
    "--username", type=str, default="bmucsanyi", help="The Weights & Biases username"
)
parser.add_argument(
    "--project", type=str, default="probit", help="The Weights & Biases project"
)
parser.add_argument(
    "--count", type=int, default=1, help="Number of runs to query from the sweep"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="imagenet",
    choices=["imagenet", "cifar10"],
    help="Dataset name",
)
parser.add_argument(
    "--simg-path",
    type=Path,
    default=Path("/mnt/lustre/work/oh/owl569/repos/probit/untangle.simg"),
    help="Path to Singularity image",
)
parser.add_argument(
    "--repo-path",
    type=Path,
    default=Path("/mnt/lustre/work/oh/owl569/repos/probit"),
    help="Path to repository",
)
parser.add_argument(
    "--datasets-root-path",
    type=Path,
    default=Path("/mnt/lustre/work/oh/owl569/datasets"),
    help="Root path of datasets",
)
parser.add_argument("--job-name", type=str, default="probit", help="Job name")
parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
parser.add_argument(
    "--ntasks-per-node", type=int, default=1, help="Number of tasks per node"
)
parser.add_argument(
    "--partition",
    type=str,
    default="2080-galvani",
    choices=["2080-galvani", "a100-galvani"],
    help="Slurm partition",
)
parser.add_argument(
    "--cpus-per-task",
    type=int,
    default=12,
    help="Number of CPUs per task",
)
parser.add_argument(
    "--mem-per-cpu", type=str, default="4G", help="Available RAM per each CPU"
)
parser.add_argument(
    "--gres", type=str, default="gpu:1", help="GPU resources to allocate per node"
)
parser.add_argument(
    "--time",
    type=str,
    default="3-00:00:00",
    help="Maximum runtime specified in the format D-HH:MM:SS",
)
parser.add_argument(
    "--log-path",
    type=Path,
    default=Path("/mnt/lustre/work/oh/owl569/logs"),
    help="The output file will be stored in this folder",
)
parser.add_argument(
    "--constraint",
    type=str,
    default=None,
    help="Target specific nodes that fulfill the constraint",
)
parser.add_argument(
    "--exclude",
    type=str,
    default=None,
    help="Exclude specific nodes",
)


class SlurmJob:
    """Submits jobs to the Slurm cluster.

    The class creates a bash script with the relevant commands, executes it and deletes
    it afterwards.

    Example:
    ``
    slurm_job = SlurmJob(
        cmd_str="python ./run.py --param=1.0",
        job_name="experiment",
        partition="2080-galvani",
        time="0-00:01:00",
        mem="8G",
    )
    slurm_job.submit()
    ``
    """

    def __init__(
        self,
        cmd_str: str,
        job_name: str,
        partition: str,
        nodes: int,
        ntasks_per_node: int,
        cpus_per_task: int,
        mem_per_cpu: str | None,
        mem: str | None,
        gres: str,
        time: str,
        log_path: Path,
        constraint: str | None,
        exclude: str | None,
    ):
        """SlurmJob constructor that stores (and check some of the) parameters.

        Args:
            cmd_str: The command line string for executing the actual program code
                (and potential setup code), e.g., `"python train.py <ARGS>"`.
            job_name: Name of the Slurm job.
            partition: This setting specifies the partition. For a complete
                list of available partitions, execute the Slurm command
                `sinfo -s`. Examples:
                cpu-galvani: 30h time limit
                2080-galvani: 3d time limit
                a100-galvani: 3d time limit
            nodes: The number of nodes.
            ntasks_per_node: The number of tasks per node.
            cpus_per_task: The number of CPUs per task.
            mem_per_cpu: Available RAM per each CPU specified in megabytes (suffix `M`)
                or gigabytes (suffix `G`).
            mem: Total available RAM in the same format as mem_per_cpu.
            gres: GPU resources to allocate. Default is `gpu:1` which allocates a single
                GPU whose type depends on the partition.
            time: Maximum runtime specified in the format D-HH:MM:SS,
                e.g., `"0-08:00:00"` for 8 hours. This needs to be compatible
                with `partition`.
            log_path: The output file will be stored in this folder.
                Default is `None`, i.e. the output and
                error file are stored in the working directory.
            constraint: With this parameter, you can target specific nodes that fulfill
                a certain constraint.
            exclude: This parameter allows to exclude certain nodes.

        Raises:
            ValueError: Either both mem_per_cpu and mem are specified or neither.
        """
        # Input checks
        self._check_job_name(job_name)

        if mem_per_cpu is not None and mem is not None:
            msg = "Both `mem_per_cpu` and `mem` are specified"
            raise ValueError(msg)

        if mem_per_cpu is None and mem is None:
            msg = "Neither `mem_per_cpu` nor `mem` is specified"
            raise ValueError(msg)

        if mem_per_cpu is not None:
            self._check_memory_format(mem_per_cpu)

        if mem is not None:
            self._check_memory_format(mem)

        self._check_time_format(time)

        # Attribute assignments
        self._cmd_str = cmd_str
        self._job_name = job_name
        self._partition = partition
        self._nodes = nodes
        self._ntasks_per_node = ntasks_per_node
        self._cpus_per_task = cpus_per_task
        self._mem_per_cpu = mem_per_cpu
        self._mem = mem
        self._gres = gres
        self._time = time
        self._log_path = log_path
        self._constraint = constraint
        self._exclude = exclude

        self._output_file_path = self._create_file_paths()

    def submit(self):
        """Submits the job to Slurm.

        Creates a temporary bash script, executes it, and finally deletes it. Note that
        the `sbatch` command returns immediately.

        Raises:
            RuntimeError: If `sbatch` command is not available.
        """
        if not self._sbatch_exists():
            msg = "No 'sbatch' command found on the system"
            raise RuntimeError(msg)

        self._log_path.mkdir(parents=True, exist_ok=True)

        bash_file_path = Path(f"{self._job_name}.sh")
        with bash_file_path.open("w") as f:
            f.write(self._create_bash_str())
        bash_file_path.chmod(0o700)

        try:
            subprocess.run(  # noqa: S603
                ["/usr/bin/sbatch", str(bash_file_path)], check=True
            )
        finally:
            bash_file_path.unlink()

    @staticmethod
    def _check_job_name(job_name):
        """Validates the job name.

        The job name must only contain letters, numbers, underscores, and hyphens.
        """
        job_name_format = r"^[a-zA-Z1-9_-]+$"

        if not re.match(job_name_format, job_name):
            msg = f"Job name '{job_name}' has incorrect format"
            raise ValueError(msg)

    @staticmethod
    def _check_memory_format(mem):
        """Validates the format of `mem` or `mem_per_cpu`.

        For example, `"13.4M"` (for 13.4 megabytes) or `"8G"` (for 8 gigabytes) are
        valid values.
        """
        mem_format = r"^(\d+)\.(\d+)[G,M]$|^(\d+)[G,M]$"

        if not re.match(mem_format, mem):
            msg = f"Memory '{mem}' has incorrect format"
            raise ValueError(msg)

    @staticmethod
    def _check_time_format(time):
        """Ensures that `time` has the right format D-HH:MM:SS."""
        time_format = r"^(\d{1})-(\d{2}):(\d{2}):(\d{2})$"

        if not re.match(time_format, time):
            msg = "Time not in format D-HH:MM:SS"
            raise ValueError(msg)

    def _create_file_paths(self):
        """Creates absolute output and error file paths based on `self.job_name`."""
        # `%j` is a placeholder for the job-id and will be filled in by Slurm
        output_path = self._log_path / f"%j_{self._job_name}.out"

        return output_path.resolve()

    def _create_sbatch_str(self):
        """Creates the configuration string that contains the `SBATCH` commands."""
        mem_option = (
            f"--mem={self._mem}"
            if self._mem is not None
            else f"--mem-per-cpu={self._mem_per_cpu}"
        )

        sbatch_str = (
            f"#SBATCH --job-name={self._job_name}\n"
            f"#SBATCH --partition={self._partition}\n"
            f"#SBATCH --nodes={self._nodes}\n"
            f"#SBATCH --ntasks-per-node={self._ntasks_per_node}\n"
            f"#SBATCH --cpus-per-task={self._cpus_per_task}\n"
            f"#SBATCH {mem_option}\n"
            f"#SBATCH --gres={self._gres}\n"
            f"#SBATCH --time={self._time}\n"
            f"#SBATCH --output={self._output_file_path}"
        )

        if self._constraint is not None:
            sbatch_str += f"\n#SBATCH --constraint={self._constraint}"

        if self._exclude is not None:
            sbatch_str += f"\n#SBATCH --exclude={self._exclude}"

        return sbatch_str

    @staticmethod
    def _create_scontrol_str():
        """Creates the `scontrol` string.

        The returned command prints important information to the output file.
        """
        return "scontrol show job $SLURM_JOB_ID"

    def _create_cmd_str(self):
        """Create the command line string for executing the actual program."""
        return self._cmd_str

    def _create_bash_str(self):
        """Creates one string that represents the content of the bash file.

        The function joins the components defined in the above methods.
        """
        bash_str = (
            f"#!/bin/bash\n\n{self._create_sbatch_str()}\n\n"
            f"{self._create_scontrol_str()}\n\n{self._create_cmd_str()}"
        )
        return bash_str

    @staticmethod
    def _sbatch_exists():
        """Check whether the `sbatch` command is available."""
        return which("sbatch") is not None


def get_cmd_str(args):
    setup_str = ""
    if args.partition == "2080-galvani":
        # Copy over data
        setup_str = "mkdir /scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets\n"
        if args.dataset == "imagenet":
            setup_str += (
                f"cp {args.datasets_root_path}/raters.npz "
                "/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets/raters.npz\n"
                f"cp {args.datasets_root_path}/real.json "
                "/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets/real.json"
            )
        else:  # "cifar10"
            setup_str += (
                f"cp -r {args.datasets_root_path}/cifar-10-batches-py "
                "/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets/"
                "cifar-10-batches-py\n"
                f"cp -r {args.datasets_root_path}/CIFAR10H "
                "/scratch_local/$SLURM_JOB_USER-$SLURM_JOBID/datasets/CIFAR10H"
            )

    # Start the wandb agent inside a singularity container
    run_str = (
        rf"singularity exec --bind /:/host --nv "
        rf'--pwd "/host/{args.repo_path}" {args.simg_path} '
        rf'bash -c "wandb agent --count {args.count} '
        rf'{args.username}/{args.project}/{getattr(args, "sweep-id")}"'
    )

    cmd_str = run_str
    if setup_str:
        cmd_str = f"{setup_str}\n\n{cmd_str}"

    return cmd_str


def main():
    args = parser.parse_args()
    cmd_str = get_cmd_str(args)

    slurm_job = SlurmJob(
        cmd_str=cmd_str,
        job_name=args.job_name,
        partition=args.partition,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks_per_node,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu=args.mem_per_cpu,
        mem=None,
        gres=args.gres,
        time=args.time,
        log_path=args.log_path,
        constraint=args.constraint,
        exclude=args.exclude,
    )
    slurm_job.submit()


if __name__ == "__main__":
    main()
