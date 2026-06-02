#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY before running this script}"
: "${GPU_QUERY:=H100}"
: "${INCLUDE_EMPTY:=0}"
: "${OUTPUT_JSON:=0}"
: "${REQUIRE_STORAGE_SUPPORT:=1}"
: "${RUNPOD_API_BASE_URL:=https://api.runpod.io}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

python3 - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

api_key = os.environ["RUNPOD_API_KEY"]
gpu_query = os.environ.get("GPU_QUERY", "H100").lower()
include_empty = os.environ.get("INCLUDE_EMPTY", "0") == "1"
output_json = os.environ.get("OUTPUT_JSON", "0") == "1"
require_storage = os.environ.get("REQUIRE_STORAGE_SUPPORT", "1") == "1"
api_base = os.environ.get("RUNPOD_API_BASE_URL", "https://api.runpod.io").rstrip("/")
fixture_path = os.environ.get("RUNPOD_DATACENTERS_JSON")

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

if fixture_path:
    data = json.loads(Path(fixture_path).read_text())
else:
    request = urllib.request.Request(
        f"{api_base}/graphql",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "nanoflow-runpod-datacenter-filter",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"RunPod GraphQL request failed: HTTP {exc.code}: {body}") from exc
    if payload.get("errors"):
        raise SystemExit(f"RunPod GraphQL errors: {payload['errors']}")
    data = payload.get("data", {}).get("dataCenters", [])

matches = []
for dc in data:
    storage_support = bool(dc.get("storageSupport"))
    if require_storage and not storage_support:
        continue
    for gpu in dc.get("gpuAvailability", []) or []:
        name = gpu.get("displayName") or gpu.get("gpuTypeDisplayName") or ""
        gpu_id = gpu.get("gpuId") or gpu.get("gpuTypeId") or ""
        haystack = f"{name} {gpu_id}".lower()
        if gpu_query not in haystack:
            continue
        stock_status = str(gpu.get("stockStatus", "") or "").strip()
        if not include_empty and not stock_status:
            continue
        matches.append({
            "id": dc.get("id", ""),
            "location": dc.get("location", ""),
            "storage_support": storage_support,
            "name": name,
            "gpu_id": gpu_id,
            "stock_status": stock_status or "Unknown",
        })

status_rank = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3, "": 3}
matches.sort(key=lambda row: (status_rank.get(row["stock_status"], 4), row["location"], row["id"], row["name"]))

if output_json:
    print(json.dumps(matches, indent=2))
    raise SystemExit

if not matches:
    print(f"No datacenters found for GPU_QUERY={gpu_query!r}.")
    if require_storage:
        print("Set REQUIRE_STORAGE_SUPPORT=0 to include data centers without network volume support.")
    if not include_empty:
        print("Set INCLUDE_EMPTY=1 to include entries with blank stockStatus.")
    raise SystemExit

print("DATA_CENTER\tLOCATION\tSTORAGE\tSTOCK\tGPU")
for row in matches:
    storage = "yes" if row["storage_support"] else "no"
    print(f"{row['id']}\t{row['location']}\t{storage}\t{row['stock_status']}\t{row['name']}")
PY
