#!/usr/bin/env python3
"""Maintain a live per-site DARCES runtime and hypothesis ledger."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import load_pipeline_config


GENERATED_PATTERN = re.compile(r"Generated\s+([0-9]+)\s+ordered hypotheses")
SCREENED_PATTERN = re.compile(r"Passed all screening:\s*([0-9]+)")
FIELDNAMES = (
    "resolution",
    "site",
    "status",
    "local_feature_count",
    "hypothesis_cap",
    "generated_hypothesis_count",
    "screened_hypothesis_count",
    "evaluated_hypothesis_count",
    "runtime_s",
    "generated_hypotheses_per_s",
    "cluster_size",
    "correspondence_count",
    "xy_error_m",
    "heading_error_deg",
    "result_source",
    "checkpoint_signature",
)
LEGACY_COUNT_CACHE: dict[str, dict[int, tuple[int | None, int | None]]] = {}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        default="all",
        help="Resolution profile or 'all' (default: all)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "results" / "diagnostics",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh repeatedly instead of writing one snapshot",
    )
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument(
        "--launcher-pid",
        type=int,
        help="With --watch, stop after this process exits",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _hypothesis_counts(log_path: Path | None) -> tuple[int | None, int | None]:
    if log_path is None or not log_path.is_file():
        return None, None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    generated = GENERATED_PATTERN.search(text)
    screened = SCREENED_PATTERN.search(text)
    return (
        int(generated.group(1)) if generated else None,
        int(screened.group(1)) if screened else None,
    )


def _legacy_hypothesis_counts(
    resolution: str,
) -> dict[int, tuple[int | None, int | None]]:
    """Recover counts from an older sequential pipeline log, if available."""
    if resolution in LEGACY_COUNT_CACHE:
        return LEGACY_COUNT_CACHE[resolution]
    log_root = PROJECT_ROOT.parent / "pipeline_logs"
    candidates = list(log_root.glob("*.log")) if log_root.is_dir() else []
    if log_root.is_dir():
        candidates.extend(log_root.glob("*/*.log"))
    candidates.extend(PROJECT_ROOT.parent.glob("*.log"))
    command_pattern = re.compile(
        rf"^>>> .*09_darces_all_sites\.py --resolution {re.escape(resolution)}\b.*$",
        re.MULTILINE,
    )
    site_pattern = re.compile(
        r"^=== Site ([0-9]+) ===\n(.*?)(?=^=== Site [0-9]+ ===|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    best: dict[int, tuple[int | None, int | None]] = {}
    best_mtime = -1
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for command in command_pattern.finditer(text):
            section = text[command.end():]
            next_command = re.search(r"^>>> ", section, re.MULTILINE)
            if next_command:
                section = section[:next_command.start()]
            counts: dict[int, tuple[int | None, int | None]] = {}
            for match in site_pattern.finditer(section):
                generated = GENERATED_PATTERN.search(match.group(2))
                screened = SCREENED_PATTERN.search(match.group(2))
                counts[int(match.group(1))] = (
                    int(generated.group(1)) if generated else None,
                    int(screened.group(1)) if screened else None,
                )
            mtime = path.stat().st_mtime_ns
            if len(counts) > len(best) or (len(counts) == len(best) and mtime > best_mtime):
                best = counts
                best_mtime = mtime
    LEGACY_COUNT_CACHE[resolution] = best
    return best


def _latest_checkpoint_run(result_directory: Path) -> Path | None:
    root = result_directory / "darces_checkpoints"
    candidates = [
        directory
        for directory in root.iterdir()
        if directory.is_dir() and (directory / "manifest.json").is_file()
    ] if root.is_dir() else []
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "manifest.json").stat().st_mtime_ns)


def _records_for_resolution(
    resolution: str,
) -> tuple[list[dict[str, object]], int, str]:
    result_directory = PROJECT_ROOT / "results" / f"{resolution}_px"
    aggregate_path = result_directory / "darces_all_sites.json"
    aggregate = _read_json(aggregate_path)
    checkpoint_run = _latest_checkpoint_run(result_directory)
    checkpoint_manifest = (
        _read_json(checkpoint_run / "manifest.json") if checkpoint_run else None
    )

    aggregate_mtime = aggregate_path.stat().st_mtime_ns if aggregate_path.is_file() else -1
    checkpoint_mtime = (
        (checkpoint_run / "manifest.json").stat().st_mtime_ns
        if checkpoint_run is not None
        else -1
    )
    use_checkpoints = checkpoint_run is not None and checkpoint_mtime > aggregate_mtime

    records: list[dict[str, object]] = []
    settings: dict[str, object] = {}
    signature = ""
    source = "none"
    log_directory: Path | None = None
    legacy_counts: dict[int, tuple[int | None, int | None]] = {}
    if use_checkpoints and checkpoint_run is not None:
        source = "checkpoint"
        settings = dict((checkpoint_manifest or {}).get("settings", {}))
        signature = str((checkpoint_manifest or {}).get("signature", checkpoint_run.name))
        log_directory = checkpoint_run / "logs"
        for path in sorted(checkpoint_run.glob("site_*.json")):
            payload = _read_json(path)
            result = (payload or {}).get("result")
            if isinstance(result, dict):
                records.append(result)
    elif aggregate is not None:
        source = "aggregate"
        settings = dict(aggregate.get("settings", {}))
        signature = str(aggregate.get("execution", {}).get("checkpoint_signature", ""))
        records = [record for record in aggregate.get("sites", []) if isinstance(record, dict)]
        legacy_counts = _legacy_hypothesis_counts(resolution)

    total_sites = len(list((PROJECT_ROOT / "local_maps" / f"{resolution}_px" / "features").glob("local_craters_site_*.npz")))
    if total_sites == 0:
        total_sites = len(records)
    hypothesis_cap = settings.get("trials_per_site")
    rows: list[dict[str, object]] = []
    for result in records:
        site = int(result["site"])
        generated, screened = _hypothesis_counts(
            log_directory / f"site_{site:04d}.log" if log_directory else None
        )
        if generated is None and screened is None:
            generated, screened = legacy_counts.get(site, (None, None))
        runtime = _finite(result.get("runtime_s"))
        rate = (
            generated / runtime
            if generated is not None and runtime is not None and runtime > 0.0
            else None
        )
        rows.append(
            {
                "resolution": resolution,
                "site": site,
                "status": result.get("status"),
                "local_feature_count": result.get("local_feature_count"),
                "hypothesis_cap": hypothesis_cap,
                "generated_hypothesis_count": generated,
                "screened_hypothesis_count": screened,
                "evaluated_hypothesis_count": result.get("evaluated_hypothesis_count"),
                "runtime_s": runtime,
                "generated_hypotheses_per_s": rate,
                "cluster_size": result.get("cluster_size"),
                "correspondence_count": result.get("correspondence_count"),
                "xy_error_m": _finite(result.get("xy_error_m")),
                "heading_error_deg": _finite(result.get("heading_error_deg")),
                "result_source": source,
                "checkpoint_signature": signature,
            }
        )
    rows.sort(key=lambda row: int(row["site"]))
    return rows, total_sites, source


def _summary_row(
    resolution: str,
    rows: list[dict[str, object]],
    total_sites: int,
    source: str,
) -> dict[str, object]:
    runtimes = [float(row["runtime_s"]) for row in rows if row["runtime_s"] is not None]
    features = [float(row["local_feature_count"]) for row in rows if row["local_feature_count"] is not None]
    generated = [float(row["generated_hypothesis_count"]) for row in rows if row["generated_hypothesis_count"] is not None]
    return {
        "resolution": resolution,
        "completed_sites": len(rows),
        "total_sites": total_sites,
        "completion_fraction": len(rows) / total_sites if total_sites else None,
        "solution_sites": sum(row["status"] == "solution" for row in rows),
        "no_solution_sites": sum(row["status"] == "no_solution" for row in rows),
        "insufficient_feature_sites": sum(
            row["status"] == "skipped_insufficient_features" for row in rows
        ),
        "runtime_samples": len(runtimes),
        "total_site_runtime_s": sum(runtimes),
        "mean_runtime_s": statistics.fmean(runtimes) if runtimes else None,
        "median_runtime_s": statistics.median(runtimes) if runtimes else None,
        "max_runtime_s": max(runtimes) if runtimes else None,
        "mean_feature_count": statistics.fmean(features) if features else None,
        "mean_generated_hypotheses": statistics.fmean(generated) if generated else None,
        "result_source": source,
    }


def write_snapshot(resolutions: tuple[str, ...], output_directory: Path) -> tuple[int, ...]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    progress: list[int] = []
    for resolution in resolutions:
        resolution_rows, total_sites, source = _records_for_resolution(resolution)
        rows.extend(resolution_rows)
        summaries.append(_summary_row(resolution, resolution_rows, total_sites, source))
        progress.append(len(resolution_rows))
        _atomic_csv(
            PROJECT_ROOT
            / "results"
            / f"{resolution}_px"
            / "darces_runtime_by_site.csv",
            resolution_rows,
            FIELDNAMES,
        )
    summary_fields = tuple(summaries[0]) if summaries else ("resolution",)
    _atomic_csv(output_directory / "darces_runtime_by_site.csv", rows, FIELDNAMES)
    _atomic_csv(output_directory / "darces_runtime_summary.csv", summaries, summary_fields)
    return tuple(progress)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    args = parse_arguments()
    if args.interval_seconds <= 0.0:
        raise ValueError("--interval-seconds must be positive")
    available = load_pipeline_config().available_resolutions
    if args.resolution == "all":
        resolutions = available
    elif args.resolution in available:
        resolutions = (args.resolution,)
    else:
        raise ValueError(
            f"Unknown resolution {args.resolution!r}; choose from "
            f"{', '.join(available)} or all"
        )

    previous: tuple[int, ...] | None = None
    while True:
        progress = write_snapshot(resolutions, args.output_directory.resolve())
        if progress != previous:
            detail = ", ".join(
                f"{resolution}={count}" for resolution, count in zip(resolutions, progress)
            )
            print(f"DARCES runtime ledger updated: {detail}", flush=True)
            previous = progress
        if not args.watch:
            break
        if args.launcher_pid is not None and not _process_exists(args.launcher_pid):
            print(f"Launcher PID {args.launcher_pid} exited; monitor stopping.", flush=True)
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
