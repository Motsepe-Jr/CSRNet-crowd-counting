from __future__ import annotations

import argparse
import os
import posixpath
import shlex
from pathlib import Path

try:
    import paramiko
except ImportError as exc:
    raise SystemExit(
        "paramiko is required for this helper. Install it in your local Python environment with: pip install paramiko"
    ) from exc


# Point these at your own cluster via the environment - there are deliberately
# no defaults, so nobody's host or account name ships with the repo:
#   CSR_CLUSTER_HOST        SSH host of the cluster login node
#   CSR_CLUSTER_USER        your username on it
#   CSR_REMOTE_PROJECT_DIR  where the repo should be staged remotely
#   CSR_CLUSTER_PASSWORD    SSH password (or use an agent / key)
# Each also has a matching --flag.
DEFAULT_HOST = ""
DEFAULT_USER = ""
DEFAULT_REMOTE_PROJECT_DIR = ""
SYNC_ROOT_FILES = [
    "train.py",
    "dataset.py",
    "image.py",
    "model.py",
    "requirements.txt",
    "utils.py",
    "wandb_utils.py",
]
SYNC_DIRS = ["scripts", "slurm_scripts"]
RAW_DATASET_DIR = "ShanghaiTech_Crowd_Counting_Dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync CSR-Net to the Wits cluster and submit the dataset->train->eval pipeline."
    )
    parser.add_argument("--host", default=os.environ.get("CSR_CLUSTER_HOST", DEFAULT_HOST))
    parser.add_argument("--user", default=os.environ.get("CSR_CLUSTER_USER", DEFAULT_USER))
    parser.add_argument(
        "--password-env",
        default="CSR_CLUSTER_PASSWORD",
        help="environment variable that stores the SSH password",
    )
    parser.add_argument(
        "--local-project-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="local CSR-Net repository root",
    )
    parser.add_argument(
        "--remote-project-dir",
        default=os.environ.get("CSR_REMOTE_PROJECT_DIR", DEFAULT_REMOTE_PROJECT_DIR),
        help="remote directory where the repo should be staged",
    )
    parser.add_argument(
        "--remote-dataset-root",
        default="",
        help="remote ShanghaiTech root; defaults to <remote-project-dir>/ShanghaiTech_Crowd_Counting_Dataset",
    )
    parser.add_argument("--part", choices=("A", "B"), default="A")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument(
        "--wandb-project",
        default="csr-net-cluster",
        help="W&B project name used by the submitted jobs",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="online",
        help="W&B mode exported to the cluster jobs",
    )
    parser.add_argument(
        "--run-stem",
        default="",
        help="stable run stem for checkpoint/output naming",
    )
    parser.add_argument(
        "--sync-dataset",
        action="store_true",
        help="also upload the raw ShanghaiTech dataset tree (excluding generated .h5 maps)",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="skip file upload and only submit the remote pipeline",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="upload files but do not submit the remote pipeline",
    )
    parser.add_argument(
        "--force-dataset",
        action="store_true",
        help="force regeneration of cluster-side density maps",
    )
    return parser


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    if not remote_dir or remote_dir == "/":
        return

    try:
        sftp.stat(remote_dir)
        return
    except OSError:
        parent = posixpath.dirname(remote_dir.rstrip("/"))
        if parent and parent != remote_dir:
            ensure_remote_dir(sftp, parent)
        try:
            sftp.mkdir(remote_dir)
        except OSError:
            sftp.stat(remote_dir)


def iter_sync_files(local_root: Path, sync_dataset: bool, part: str):
    for relative_path in SYNC_ROOT_FILES:
        local_path = local_root / relative_path
        if not local_path.exists():
            raise FileNotFoundError(local_path)
        yield local_path, relative_path

    for relative_dir in SYNC_DIRS:
        base_dir = local_root / relative_dir
        if not base_dir.exists():
            continue
        for local_path in sorted(base_dir.rglob("*")):
            if local_path.is_file() and "__pycache__" not in local_path.parts and local_path.suffix != ".pyc":
                yield local_path, local_path.relative_to(local_root).as_posix()

    if sync_dataset:
        dataset_root = local_root / RAW_DATASET_DIR
        if not dataset_root.exists():
            raise FileNotFoundError(dataset_root)
        selected_part_dir = dataset_root / f"part_{part.upper()}_final"
        if not selected_part_dir.exists():
            raise FileNotFoundError(selected_part_dir)
        for local_path in sorted(selected_part_dir.rglob("*")):
            if local_path.is_file() and local_path.suffix.lower() != ".h5":
                yield local_path, local_path.relative_to(local_root).as_posix()


def upload_tree(
    sftp: paramiko.SFTPClient,
    *,
    local_root: Path,
    remote_root: str,
    sync_dataset: bool,
    part: str,
) -> None:
    ensure_remote_dir(sftp, remote_root)
    for local_path, relative_path in iter_sync_files(local_root, sync_dataset, part):
        remote_path = posixpath.join(remote_root, relative_path)
        ensure_remote_dir(sftp, posixpath.dirname(remote_path))
        print(f"Uploading {relative_path}")
        sftp.put(str(local_path), remote_path)
        if remote_path.endswith(".sh"):
            sftp.chmod(remote_path, 0o755)


def build_remote_submit_command(args: argparse.Namespace) -> str:
    remote_dataset_root = args.remote_dataset_root or posixpath.join(
        args.remote_project_dir, RAW_DATASET_DIR
    )
    run_stem = args.run_stem or f"csrnet_part{args.part.lower()}"
    env_vars = {
        "PROJECT_DIR": args.remote_project_dir,
        "DATASET_ROOT": remote_dataset_root,
        "PART": args.part,
        "EPOCHS": str(args.epochs),
        "BATCH_SIZE": str(args.batch_size),
        "WORKERS": str(args.workers),
        "LR": str(args.lr),
        "VAL_RATIO": str(args.val_ratio),
        "SEED": str(args.seed),
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_MODE": args.wandb_mode,
        "RUN_STEM": run_stem,
        "FORCE_REBUILD": "true" if args.force_dataset else "false",
    }
    if args.train_limit is not None:
        env_vars["TRAIN_LIMIT"] = str(args.train_limit)
    if args.test_limit is not None:
        env_vars["TEST_LIMIT"] = str(args.test_limit)

    wandb_api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if wandb_api_key:
        env_vars["WANDB_API_KEY"] = wandb_api_key

    command_parts = [f"cd {shlex.quote(args.remote_project_dir)}"]
    for key, value in env_vars.items():
        command_parts.append(f"export {key}={shlex.quote(value)}")
    command_parts.append("bash slurm_scripts/10_submit_csrnet_pipeline.sh")
    return "; ".join(command_parts)


def main() -> None:
    args = build_parser().parse_args()

    missing = [
        name
        for name, value in (
            ("--host / CSR_CLUSTER_HOST", args.host),
            ("--user / CSR_CLUSTER_USER", args.user),
            ("--remote-project-dir / CSR_REMOTE_PROJECT_DIR", args.remote_project_dir),
        )
        if not value
    ]
    if missing:
        raise SystemExit("cluster target not configured; set " + ", ".join(missing))

    local_root = Path(args.local_project_dir).resolve()
    password = os.environ.get(args.password_env, "")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=args.host, username=args.user, password=password or None)
    try:
        if not args.skip_sync:
            with client.open_sftp() as sftp:
                upload_tree(
                    sftp,
                    local_root=local_root,
                    remote_root=args.remote_project_dir,
                    sync_dataset=args.sync_dataset,
                    part=args.part,
                )

        if args.sync_only:
            print("Sync completed; remote submission was skipped.")
            return

        remote_command = build_remote_submit_command(args)
        print(f"Running remote command: {remote_command}")
        stdin, stdout, stderr = client.exec_command(remote_command)
        exit_code = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode("utf-8", errors="replace")
        stderr_text = stderr.read().decode("utf-8", errors="replace")
        if stdout_text:
            print(stdout_text.rstrip())
        if stderr_text:
            print(stderr_text.rstrip())
        if exit_code != 0:
            raise SystemExit(exit_code)
    finally:
        client.close()


if __name__ == "__main__":
    main()