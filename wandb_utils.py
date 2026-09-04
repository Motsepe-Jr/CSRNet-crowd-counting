from __future__ import annotations

import argparse
import os
from typing import Any

try:
    import wandb
except ImportError:
    wandb = None


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
        help="Weights & Biases mode; default keeps training fully local.",
    )
    parser.add_argument("--wandb-project", default="", help="W&B project name")
    parser.add_argument("--wandb-entity", default="", help="W&B entity or team name")
    parser.add_argument("--wandb-group", default="", help="W&B run group")
    parser.add_argument("--wandb-run-name", default="", help="W&B run name")
    parser.add_argument(
        "--wandb-tags",
        default="",
        help="comma-separated W&B tags",
    )


def _parse_tags(raw_tags: str) -> list[str]:
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def init_wandb(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    default_project: str,
    job_type: str,
):
    if getattr(args, "wandb_mode", "disabled") == "disabled":
        return None
    if wandb is None:
        raise RuntimeError(
            "wandb logging was requested, but the wandb package is not installed."
        )

    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if api_key:
        wandb.login(key=api_key, relogin=True)

    project = (
        getattr(args, "wandb_project", "")
        or os.environ.get("WANDB_PROJECT", "").strip()
        or default_project
    )
    init_kwargs: dict[str, Any] = {
        "project": project,
        "config": config,
        "job_type": job_type,
        "mode": getattr(args, "wandb_mode", "disabled"),
    }
    if getattr(args, "wandb_entity", ""):
        init_kwargs["entity"] = args.wandb_entity
    if getattr(args, "wandb_group", ""):
        init_kwargs["group"] = args.wandb_group
    if getattr(args, "wandb_run_name", ""):
        init_kwargs["name"] = args.wandb_run_name

    tags = _parse_tags(getattr(args, "wandb_tags", ""))
    if tags:
        init_kwargs["tags"] = tags

    return wandb.init(**init_kwargs)