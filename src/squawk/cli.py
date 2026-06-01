from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from squawk.config import RuntimeConfig, VerifyLevel, load_config
from squawk.constants import SWIFT_ENDPOINT
from squawk.download import FetchResult, run_mirror
from squawk.manifest import Manifest, ManifestEntry
from squawk.merge import merge_source
from squawk.models import RemoteObject
from squawk.pack import pack_source
from squawk.segment import segment_source
from squawk.sources import get_source, iter_sources
from squawk.sources.base import Source
from squawk.swift import SwiftClient

app = typer.Typer(
    name="squawk",
    help="Faithful resumable mirror + merge for paired ATC-audio / ADS-B datasets.",
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
)

_err = Console(stderr=True)
_out = Console()

MANIFEST_FILENAME = ".mirror_manifest.jsonl"
_BYTES_PER_GB = 1e9


class Location(StrEnum):
    KAGC = "kagc"
    KBTP = "kbtp"
    ALL = "all"


class Kind(StrEnum):
    AUDIO = "audio"
    ADSB = "adsb"
    BOTH = "both"


@app.command()
def sources() -> None:
    """List registered sources with their license and attribution (offline)."""
    for source in iter_sources():
        _out.print(f"[bold]{source.name}[/bold]  ({source.license})")
        _out.print(f"  {source.description}")
        _out.print(f"  {source.attribution}")


@app.command()
def prepare(
    source: Annotated[
        str, typer.Argument(help="Registered source, e.g. tartanaviation")
    ],
    location: Annotated[Location, typer.Option(help="Airport filter")] = Location.ALL,
    kind: Annotated[
        Kind, typer.Option(help="Which container(s) to mirror")
    ] = Kind.BOTH,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="HEAD-sweep only; transfer nothing")
    ] = False,
    verify: Annotated[VerifyLevel, typer.Option(help="Completeness tier")] = "size",
    verify_only: Annotated[
        bool,
        typer.Option(
            "--verify-only",
            help="Check the existing mirror against the source; no download",
        ),
    ] = False,
    workers: Annotated[int | None, typer.Option(help="Concurrency cap")] = None,
    store: Annotated[Path | None, typer.Option(help="Mirror root override")] = None,
) -> None:
    """Mirror and verify a source's audio and ADS-B objects."""
    cfg = _config_for(location, verify, workers, store)
    src = _get_source(source)
    client = _build_client(cfg)
    objects = list(_enumerate(src, cfg, kind))

    if dry_run:
        _report_plan(client, objects, cfg.mirror_root)
        return

    if verify_only:
        _verify_mirror(client, objects, cfg.mirror_root)
        return

    _run(client, objects, cfg)


@app.command()
def merge(
    source: Annotated[
        str, typer.Argument(help="Registered source, e.g. tartanaviation")
    ],
    location: Annotated[Location, typer.Option(help="Airport filter")] = Location.ALL,
    date_range: Annotated[
        str | None,
        typer.Option(
            "--date-range", help="Inclusive MM-DD-YY range, e.g. 10-01-21,10-31-21"
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Parquet output dir (default <mirror_root>/parquet/clips)"),
    ] = None,
    workers: Annotated[int | None, typer.Option(help="Partition concurrency")] = None,
    store: Annotated[Path | None, typer.Option(help="Mirror root override")] = None,
) -> None:
    """Window-join mirrored audio clips to ADS-B tracks → Stage-1 parquet."""
    _get_source(source)
    cfg = _merge_config(location, date_range, workers, store)
    out_dir = out if out is not None else cfg.mirror_root / "parquet" / "clips"

    stats = merge_source(source, cfg, out_dir, max_workers=workers)

    _report_merge(stats, out_dir)
    if stats["clips"] == 0:
        raise typer.Exit(code=1)


def _merge_config(
    location: Location, date_range: str | None, workers: int | None, store: Path | None
) -> RuntimeConfig:
    overrides: dict[str, object] = {}
    if location is not Location.ALL:
        overrides["airports"] = (location.value,)
    if date_range is not None:
        overrides["date_range"] = date_range
    if workers is not None:
        overrides["max_workers"] = workers
    if store is not None:
        overrides["mirror_root"] = store
    return load_config(overrides=overrides)


def _report_merge(stats: dict, out_dir: Path) -> None:
    clips = stats["clips"]
    pct = 100 * stats["with_adsb"] / clips if clips else 0.0
    _out.print(
        f"Merged {clips} clips across {stats['partitions']} partitions"
        f" ({pct:.0f}% with ADS-B) -> {out_dir}"
    )
    if stats.get("failed"):
        _out.print(f"[red]{stats['failed']} partition(s) failed to parse[/red]")


@app.command()
def pack(
    source: Annotated[
        str, typer.Argument(help="Registered source, e.g. tartanaviation")
    ],
    in_dir: Annotated[
        Path | None,
        typer.Option(
            "--in", help="Stage-1 clips dir (default <mirror_root>/parquet/clips)"
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Packed output dir (default <mirror_root>/parquet/packed)"),
    ] = None,
    max_shard_mb: Annotated[
        int, typer.Option("--max-shard-mb", help="Shard cap by uncompressed audio MB")
    ] = 250,
    sample_rate: Annotated[
        int, typer.Option("--sample-rate", help="Target resample rate (Hz)")
    ] = 16000,
    workers: Annotated[int | None, typer.Option(help="Worker concurrency")] = None,
    store: Annotated[Path | None, typer.Option(help="Mirror root override")] = None,
) -> None:
    """Embed resampled 16 kHz WAV bytes into HF-friendly sharded parquet → Stage-2."""
    _get_source(source)
    cfg = load_config(overrides=_store_override(store))
    clips_dir = in_dir if in_dir is not None else cfg.mirror_root / "parquet" / "clips"
    out_dir = out if out is not None else cfg.mirror_root / "parquet" / "packed"

    stats = pack_source(
        clips_dir,
        out_dir,
        cfg.mirror_root,
        max_shard_mb=max_shard_mb,
        sample_rate=sample_rate,
        max_workers=workers,
    )

    _report_pack(stats, out_dir)
    if stats["clips"] == 0:
        raise typer.Exit(code=1)


def _store_override(store: Path | None) -> dict[str, object]:
    return {"mirror_root": store} if store is not None else {}


def _report_pack(stats: dict, out_dir: Path) -> None:
    mb = stats["bytes"] / 1e6
    _out.print(
        f"Packed {stats['clips']} clips into {stats['shards']} shards"
        f" ({mb:.0f} MB audio) -> {out_dir}"
    )


@app.command()
def segment(
    source: Annotated[
        str,
        typer.Argument(help="Packed parquet dir or HF repo id (e.g. owner/name)"),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Utterance output dir"),
    ] = Path("parquet/utterances"),
    max_shard_mb: Annotated[
        int, typer.Option("--max-shard-mb", help="Shard cap by uncompressed audio MB")
    ] = 250,
    sample_rate: Annotated[
        int, typer.Option("--sample-rate", help="Embedded audio rate (Hz)")
    ] = 16000,
    device: Annotated[
        str, typer.Option("--device", help="Torch device for the VAD segmenter")
    ] = "cuda",
) -> None:
    """VAD-split each packed clip into utterances → embedded sharded parquet → Stage-3."""
    stats = segment_source(
        source,
        out,
        max_shard_mb=max_shard_mb,
        sample_rate=sample_rate,
        device=device,
    )

    _report_segment(stats, out)
    if stats["clips"] == 0:
        raise typer.Exit(code=1)


def _report_segment(stats: dict, out_dir: Path) -> None:
    mb = stats["bytes"] / 1e6
    _out.print(
        f"Segmented {stats['clips']} clips into {stats['utterances']} utterances"
        f" across {stats['shards']} shards ({mb:.0f} MB audio) -> {out_dir}"
    )


def _get_source(name: str) -> Source:
    try:
        return get_source(name)
    except ValueError as error:
        _err.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error


def _config_for(
    location: Location, verify: VerifyLevel, workers: int | None, store: Path | None
) -> RuntimeConfig:
    overrides: dict[str, object] = {"verify_level": verify}
    if location is not Location.ALL:
        overrides["airports"] = (location.value,)
    if workers is not None:
        overrides["max_workers"] = workers
    if store is not None:
        overrides["mirror_root"] = store
    return load_config(overrides=overrides)


def _build_client(cfg: RuntimeConfig) -> SwiftClient:
    limits = httpx.Limits(
        max_connections=cfg.max_workers, max_keepalive_connections=cfg.max_workers
    )
    http = httpx.Client(verify=cfg.tls_verify, limits=limits)
    return SwiftClient(SWIFT_ENDPOINT, client=http)


def _enumerate(source: Source, cfg: RuntimeConfig, kind: Kind) -> list[RemoteObject]:
    objects: list[RemoteObject] = []
    if kind in (Kind.AUDIO, Kind.BOTH):
        objects += source.audio_objects(cfg)
    if kind in (Kind.ADSB, Kind.BOTH):
        objects += source.adsb_objects(cfg)
    return objects


def _report_plan(
    client: SwiftClient, objects: list[RemoteObject], mirror_root: Path
) -> None:
    pending_bytes = 0
    pending_count = 0
    for obj in objects:
        size = client.stat(obj.container, obj.key).size
        if not _already_mirrored(mirror_root / obj.rel_path, size):
            pending_count += 1
            pending_bytes += size
    gb = pending_bytes / _BYTES_PER_GB
    _out.print(f"Will download {pending_count} of {len(objects)} objects, {gb:.2f} GB")


def _already_mirrored(dest: Path, size: int) -> bool:
    return dest.exists() and dest.stat().st_size == size


class VerifyStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class VerifyResult:
    key: str
    status: VerifyStatus
    expected: int
    actual: int


def _verify_object(
    client: SwiftClient, obj: RemoteObject, mirror_root: Path
) -> VerifyResult:
    expected = client.stat(obj.container, obj.key).size
    dest = mirror_root / obj.rel_path
    if not dest.exists():
        return VerifyResult(obj.key, VerifyStatus.MISSING, expected, 0)
    actual = dest.stat().st_size
    if actual != expected:
        return VerifyResult(obj.key, VerifyStatus.MISMATCH, expected, actual)
    return VerifyResult(obj.key, VerifyStatus.OK, expected, actual)


def _verify_mirror(
    client: SwiftClient, objects: list[RemoteObject], mirror_root: Path
) -> None:
    results = [_verify_object(client, obj, mirror_root) for obj in objects]
    problems = [r for r in results if r.status is not VerifyStatus.OK]
    complete = len(results) - len(problems)

    _out.print(f"Verified {complete} of {len(results)} objects against the source")
    for problem in problems:
        _out.print(
            f"[red]{problem.status.value.upper()} {problem.key}"
            f" (expected {problem.expected}, found {problem.actual})[/red]"
        )
    _report_reconcile(client, objects)

    if problems:
        raise typer.Exit(code=1)


def _report_reconcile(client: SwiftClient, objects: list[RemoteObject]) -> None:
    # Reconcile against the listing, not X-Container-Object-Count: that header is
    # unreliable on the CMU endpoint (observed 0 and 16312 for a 695-object container).
    for container in sorted({obj.container for obj in objects}):
        selected = sum(1 for obj in objects if obj.container == container)
        listed = len(client.list_container(container))
        _out.print(f"  {container}: selected {selected} of {listed} objects listed")


def _run(client: SwiftClient, objects: list[RemoteObject], cfg: RuntimeConfig) -> None:
    manifest = Manifest(cfg.mirror_root / MANIFEST_FILENAME)
    with _mirror_progress(len(objects)) as advance:

        def on_result(result: FetchResult) -> None:
            manifest.append(
                ManifestEntry(
                    key=result.key,
                    status=result.status,
                    size=result.size,
                    etag=result.etag,
                )
            )
            advance()

        results = run_mirror(
            client,
            objects,
            cfg.mirror_root,
            max_workers=cfg.max_workers,
            on_result=on_result,
        )
    failures = [result for result in results if result.status == "fail"]
    _summarize(results, failures)
    if failures:
        raise typer.Exit(code=1)


def _summarize(results: list[FetchResult], failures: list[FetchResult]) -> None:
    counts = Counter(result.status for result in results)
    _err.print(f"Done: ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}")
    for failure in failures:
        _err.print(f"[red]FAIL {failure.key}[/red]")


@contextmanager
def _mirror_progress(total: int) -> Iterator[Callable[[], None]]:
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=_err,
        disable=not sys.stdout.isatty(),
    )
    with progress:
        task = progress.add_task("mirroring", total=total)
        yield lambda: progress.advance(task)
