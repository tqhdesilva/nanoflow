#!/usr/bin/env python3
"""Render and launch a RunPod SkyPilot chain for NanoFlow."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_DATASET_CLOUD_URI = "gs://nanoflow/imagenet256/latent/sd-vae-ft-ema-mmap"
DEFAULT_ARTIFACT_CLOUD_URI = "gs://nanoflow/runs"
DEFAULT_IMAGE_ID = "docker:ghcr.io/tqhdesilva/nanoflow:runpod-cu124"
DEFAULT_GPU_REQUEST = "H100-SXM:1"
DEFAULT_VOLUME_NAME = "nf-imagenet256"
DEFAULT_VOLUME_SIZE = "100Gi"
DEFAULT_PREFERRED_DATACENTER = "EU-NL-1"
RUNPOD_API_BASE_URL = "https://api.runpod.io"
PLACEHOLDER_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
STOCK_RANK = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3, "": 3}
SUPPORTED_SKYPILOT_RUNPOD_COUNTRIES = {"CA", "CZ", "IS", "NL", "NO", "RO", "SE", "US"}


@dataclass(frozen=True)
class RuntimeValues:
    """Launch-time values used to render and submit a chain YAML.

    Args:
        chain_id: Stable identity for all stages in one chain.
        runpod_infra: Concrete SkyPilot RunPod infra string.
        gpu_request: SkyPilot accelerator request, such as `H100-SXM:1`.
        volume_name: RunPod network volume name mounted at `/workspace`.
        volume_size: Requested network volume size for `sky volumes apply`.
        image_id: Docker image for GPU train and eval tasks.
        sync_image_id: Docker image for CPU sync tasks.
        dataset_cloud_uri: Durable GCS source for latent mmap data.
        artifact_cloud_uri: Durable GCS sink for chain artifacts.
        chain_template_path: Template path used to produce the rendered chain.
        rendered_path: Local path for the rendered chain YAML submitted to SkyPilot.
        gcp_credentials: Optional service account JSON path.
    """

    chain_id: str
    runpod_infra: str
    gpu_request: str
    volume_name: str
    volume_size: str
    image_id: str
    sync_image_id: str
    dataset_cloud_uri: str
    artifact_cloud_uri: str
    chain_template_path: str
    rendered_path: Path
    gcp_credentials: Path | None = None

    def template_mapping(self) -> dict[str, str]:
        """Return string values for template substitution."""
        return {
            "CHAIN_ID": self.chain_id,
            "RUNPOD_INFRA": self.runpod_infra,
            "GPU_REQUEST": self.gpu_request,
            "VOLUME_NAME": self.volume_name,
            "VOLUME_SIZE": self.volume_size,
            "IMAGE_ID": self.image_id,
            "SYNC_IMAGE_ID": self.sync_image_id,
            "DATASET_CLOUD_URI": self.dataset_cloud_uri,
            "ARTIFACT_CLOUD_URI": self.artifact_cloud_uri,
            "CHAIN_TEMPLATE_PATH": self.chain_template_path,
            "RENDERED_CHAIN_PATH": str(self.rendered_path),
        }


def render_template_text(text: str, values: Mapping[str, str]) -> str:
    """Replace `${NAME}` placeholders with supplied launch-time values.

    Shell parameter expressions such as `${VAR:-fallback}` are intentionally not
    matched, so task scripts can still use Bash defaults.
    """

    missing = sorted(
        {name for name in PLACEHOLDER_RE.findall(text) if name not in values}
    )
    if missing:
        raise ValueError(f"Missing template values: {', '.join(missing)}")

    def replace(match: re.Match[str]) -> str:
        return values[match.group(1)]

    return PLACEHOLDER_RE.sub(replace, text)


def resolve_runtime_values(
    args: argparse.Namespace,
    env: Mapping[str, str] | None = None,
    *,
    dry_run: bool = False,
) -> RuntimeValues:
    """Resolve CLI args and environment defaults into launch values.

    Args:
        args: Parsed command line arguments.
        env: Environment mapping used for defaults.
        dry_run: When true, `--infra auto` can fall back to the preferred
            data center if `RUNPOD_API_KEY` is absent, because no resources are
            created.

    Returns:
        Fully resolved launch values for rendering and optional launch.
    """
    env = env or os.environ
    template_path = Path(args.template)
    chain_template_path = _remote_template_path(template_path)
    chain_id = args.chain_id or env.get("CHAIN_ID") or generate_chain_id()
    runpod_infra = args.infra or env.get("RUNPOD_INFRA") or "auto"
    gpu_request = args.gpu_request or env.get("GPU_REQUEST") or DEFAULT_GPU_REQUEST
    if runpod_infra == "auto":
        api_key = env.get("RUNPOD_API_KEY")
        preferred_datacenter = env.get(
            "RUNPOD_PREFERRED_DATACENTER", DEFAULT_PREFERRED_DATACENTER
        )
        if dry_run and not api_key:
            runpod_infra = f"runpod/{_country_from_datacenter_id(preferred_datacenter)}/{preferred_datacenter}"
        else:
            runpod_infra = select_runpod_infra(
                gpu_request,
                api_key=api_key,
                preferred_datacenter=preferred_datacenter,
            )
    volume_name = args.volume_name or env.get("VOLUME_NAME") or DEFAULT_VOLUME_NAME
    volume_size = args.volume_size or env.get("VOLUME_SIZE") or DEFAULT_VOLUME_SIZE
    image_id = args.image_id or env.get("IMAGE_ID") or DEFAULT_IMAGE_ID
    sync_image_id = args.sync_image_id or env.get("SYNC_IMAGE_ID") or image_id
    dataset_cloud_uri = (
        args.dataset_cloud_uri
        or env.get("DATASET_CLOUD_URI")
        or env.get("DATASET_GCS_URI")
        or DEFAULT_DATASET_CLOUD_URI
    )
    artifact_cloud_uri = (
        args.artifact_cloud_uri
        or env.get("ARTIFACT_CLOUD_URI")
        or DEFAULT_ARTIFACT_CLOUD_URI
    )
    credentials = _resolve_credentials(args.gcp_credentials, env)
    rendered_path = (
        Path(args.output) if args.output else _default_rendered_path(chain_id)
    )
    return RuntimeValues(
        chain_id=chain_id,
        runpod_infra=runpod_infra,
        gpu_request=gpu_request,
        volume_name=volume_name,
        volume_size=volume_size,
        image_id=image_id,
        sync_image_id=sync_image_id,
        dataset_cloud_uri=dataset_cloud_uri,
        artifact_cloud_uri=artifact_cloud_uri,
        chain_template_path=chain_template_path,
        rendered_path=rendered_path,
        gcp_credentials=credentials,
    )


def generate_chain_id(prefix: str = "chain") -> str:
    """Generate a sortable chain id."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def select_runpod_infra(
    gpu_request: str,
    *,
    api_key: str | None,
    preferred_datacenter: str = DEFAULT_PREFERRED_DATACENTER,
) -> str:
    """Select a RunPod data center with network volumes and GPU stock."""
    if not api_key:
        raise ValueError(
            "RUNPOD_API_KEY is required when --infra is auto. "
            "Pass --infra runpod/<country>/<data-center> to skip auto selection."
        )
    datacenters = fetch_runpod_datacenters(api_key)
    return select_runpod_infra_from_datacenters(
        datacenters,
        gpu_request=gpu_request,
        preferred_datacenter=preferred_datacenter,
    )


def fetch_runpod_datacenters(api_key: str) -> list[dict[str, Any]]:
    """Fetch RunPod data center stock details from GraphQL."""
    query = """
query getAllDatacenters {
  dataCenters {
    id
    name
    location
    storageSupport
    gpuAvailability {
      stockStatus
      gpuTypeId
      gpuTypeDisplayName
      displayName
    }
  }
}
"""
    request = urllib.request.Request(
        f"{RUNPOD_API_BASE_URL}/graphql",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "nanoflow-runpod-chain-launcher",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod GraphQL HTTP {exc.code}: {body}") from exc
    if payload.get("errors"):
        raise RuntimeError(f"RunPod GraphQL errors: {payload['errors']}")
    return list(payload.get("data", {}).get("dataCenters", []))


def select_runpod_infra_from_datacenters(
    datacenters: Sequence[Mapping[str, Any]],
    *,
    gpu_request: str,
    preferred_datacenter: str = DEFAULT_PREFERRED_DATACENTER,
) -> str:
    """Choose a concrete RunPod infra string from data center records."""
    matches = _matching_datacenters(datacenters, gpu_request)
    matches = [
        row
        for row in matches
        if _country_from_datacenter_id(row["id"]) in SUPPORTED_SKYPILOT_RUNPOD_COUNTRIES
    ]
    if not matches:
        raise ValueError(
            f"No SkyPilot-supported RunPod data center has storage support and stock for {gpu_request!r}"
        )
    preferred = [row for row in matches if row["id"] == preferred_datacenter]
    chosen = preferred[0] if preferred else matches[0]
    country = _country_from_datacenter_id(chosen["id"])
    return f"runpod/{country}/{chosen['id']}"


def build_volume_apply_command(values: RuntimeValues) -> list[str]:
    """Build the idempotent RunPod network volume apply command."""
    return [
        "sky",
        "volumes",
        "apply",
        "--infra",
        values.runpod_infra,
        "--name",
        values.volume_name,
        "--type",
        "runpod-network-volume",
        "--size",
        values.volume_size,
        "-y",
    ]


def build_jobs_launch_command(
    values: RuntimeValues,
    secret_path: Path | None = None,
) -> list[str]:
    """Build the SkyPilot managed chain launch command without side effects."""
    command = [
        "sky",
        "jobs",
        "launch",
        "-n",
        f"nf-imagenet256-dit-{values.chain_id}",
    ]
    command.append("--detach-run")
    if values.gcp_credentials is not None and secret_path is not None:
        command.extend(["--secret-file", str(secret_path)])
    command.extend(["-y", str(values.rendered_path)])
    return command


def render_chain_file(template_path: Path, values: RuntimeValues) -> str:
    """Render a chain template to `values.rendered_path` and return the text."""
    rendered = render_template_text(
        template_path.read_text(), values.template_mapping()
    )
    values.rendered_path.parent.mkdir(parents=True, exist_ok=True)
    values.rendered_path.write_text(rendered)
    return rendered


def run_command(command: Sequence[str]) -> None:
    """Run a command with inherited stdio and redacted failure output."""
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            f"{redacted_command(command)}"
        )


def redacted_command(command: Sequence[str]) -> str:
    """Return a shell-ish command string with inline secret values redacted."""
    redacted: list[str] = []
    skip_next_secret = False
    for item in command:
        if skip_next_secret:
            redacted.append(_redact_secret_arg(item))
            skip_next_secret = False
            continue
        redacted.append(item)
        if item == "--secret":
            skip_next_secret = True
    return " ".join(_quote(part) for part in redacted)


def _matching_datacenters(
    datacenters: Sequence[Mapping[str, Any]], gpu_request: str
) -> list[dict[str, str]]:
    gpu_query = gpu_request.split(":", 1)[0].lower().replace("-", " ")
    compact_query = gpu_query.replace(" ", "")
    matches: list[dict[str, str]] = []
    for datacenter in datacenters:
        if not bool(datacenter.get("storageSupport")):
            continue
        datacenter_id = str(datacenter.get("id", ""))
        for gpu in datacenter.get("gpuAvailability", []) or []:
            display = str(
                gpu.get("displayName")
                or gpu.get("gpuTypeDisplayName")
                or gpu.get("gpuTypeId")
                or ""
            )
            stock_status = str(gpu.get("stockStatus") or "").strip()
            haystack = display.lower().replace("-", " ")
            compact_haystack = haystack.replace(" ", "")
            if gpu_query not in haystack and compact_query not in compact_haystack:
                continue
            if stock_status not in {"High", "Medium", "Low"}:
                continue
            matches.append(
                {
                    "id": datacenter_id,
                    "location": str(datacenter.get("location", "")),
                    "gpu": display,
                    "stock_status": stock_status,
                }
            )
    matches.sort(
        key=lambda row: (
            STOCK_RANK.get(row["stock_status"], 4),
            row["location"],
            row["id"],
            row["gpu"],
        )
    )
    return matches


def _country_from_datacenter_id(datacenter_id: str) -> str:
    parts = datacenter_id.split("-")
    if len(parts) >= 2 and parts[0] in {"EU", "EUR"}:
        return parts[1]
    if parts and parts[0]:
        return parts[0]
    raise ValueError(f"Cannot infer country from RunPod data center {datacenter_id!r}")


def write_secret_file(values: RuntimeValues) -> Path:
    """Write a SkyPilot dotenv secret file under the system temp directory."""
    if values.gcp_credentials is None:
        raise ValueError("gcp_credentials is required to write a secret file")
    secret_dir = _secret_file_path(values).parent
    secret_dir.mkdir(parents=True, exist_ok=True)
    fd, secret_name = tempfile.mkstemp(
        prefix=f"{values.chain_id}.",
        suffix=".secrets.env",
        dir=secret_dir,
    )
    secret_path = Path(secret_name)
    encoded = base64.b64encode(values.gcp_credentials.read_bytes()).decode("ascii")
    with os.fdopen(fd, "w") as handle:
        handle.write(f"GCP_SERVICE_ACCOUNT_JSON_B64={encoded}\n")
    return secret_path


def _secret_file_path(values: RuntimeValues) -> Path:
    return Path(tempfile.gettempdir()) / "nanoflow-runpod-chain" / f"{values.chain_id}.secrets.env"


def _remote_template_path(template_path: Path) -> str:
    """Return a stable template path for metadata and substitutions."""
    resolved = template_path.expanduser().resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return str(resolved)


def _resolve_credentials(value: str | None, env: Mapping[str, str]) -> Path | None:
    """Resolve the optional GCP credential file from args or environment."""
    path_value = value or env.get("GOOGLE_APPLICATION_CREDENTIALS")
    if path_value is None:
        default = Path.home() / ".config" / "gcp" / "nanoflow-gcs-reader.json"
        path = default if default.exists() else None
    else:
        path = Path(path_value).expanduser()
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"GCP credentials file not found: {path}")
    return path


def _default_rendered_path(chain_id: str) -> Path:
    return Path(".nanoflow_runpod_chains") / f"{chain_id}.yaml"


def _redact_secret_arg(value: str) -> str:
    if "=" not in value:
        return "<redacted>"
    name, _ = value.split("=", 1)
    return f"{name}=<redacted>"


def _quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+@,-]+", value):
        return value
    return json.dumps(value)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path, help="SkyPilot chain YAML template")
    parser.add_argument(
        "--dry-run", action="store_true", help="render but do not launch"
    )
    parser.add_argument("--output", type=Path, help="rendered YAML output path")
    parser.add_argument("--chain-id", help="stable run identity for the chain")
    parser.add_argument("--infra", help="RunPod infra, or auto")
    parser.add_argument("--gpu-request", help="GPU request, such as H100-SXM:1")
    parser.add_argument("--volume-name", help="RunPod network volume name")
    parser.add_argument("--volume-size", help="RunPod network volume size")
    parser.add_argument("--image-id", help="GPU task image id")
    parser.add_argument("--sync-image-id", help="CPU sync task image id")
    parser.add_argument("--dataset-cloud-uri", help="GCS latent cache URI")
    parser.add_argument("--artifact-cloud-uri", help="GCS artifact output URI")
    parser.add_argument("--gcp-credentials", help="service account JSON file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the chain, apply the volume, and launch unless dry-run is set."""
    args = _parse_args(argv)
    values = resolve_runtime_values(args, dry_run=args.dry_run)
    render_chain_file(Path(args.template), values)
    volume_command = build_volume_apply_command(values)
    print(f"Rendered chain: {values.rendered_path}")
    print(f"Volume command: {redacted_command(volume_command)}")
    if args.dry_run:
        launch_command = build_jobs_launch_command(values)
        print(f"Launch command: {redacted_command(launch_command)}")
        if values.gcp_credentials is not None:
            print("Secret file: created temporarily only during a real launch")
        return 0
    run_command(volume_command)
    secret_path = None
    try:
        if values.gcp_credentials is not None:
            secret_path = write_secret_file(values)
        launch_command = build_jobs_launch_command(values, secret_path=secret_path)
        print(f"Launch command: {redacted_command(launch_command)}")
        run_command(launch_command)
    finally:
        if secret_path is not None:
            secret_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
