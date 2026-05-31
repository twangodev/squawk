# squawk

Faithful, resumable, verifiable mirror (and later merge) for paired ATC-audio / ADS-B
datasets. The first source is CMU **TartanAviation**.

`squawk` mirrors paired ATC-audio and ADS-B corpora to a deterministic local tree —
byte-for-byte, resumably (HTTP Range), and verifiably (completeness checked against the
source store) — behind a `Source` Protocol and a source-agnostic httpx download engine.

## Install

The PyPI distribution is **`squawk-atc`** (the name `squawk` was taken); the import package
and CLI are `squawk`.

```bash
pip install squawk-atc          # then: squawk ...
# or run without installing:
uvx --from squawk-atc squawk sources
```

## Usage

```bash
squawk sources                                    # registered sources + license/attribution (offline)
squawk prepare tartanaviation                     # mirror + verify the corpus
squawk prepare tartanaviation --kind adsb --dry-run   # plan only; transfer nothing
squawk prepare tartanaviation --verify-only       # check an existing mirror against the store
squawk merge tartanaviation                       # stub (audio<->ADS-B join deferred)
```

Key `prepare` flags: `--location {kagc,kbtp,all}`, `--kind {audio,adsb,both}`, `--dry-run`,
`--verify {size,sha256,none}`, `--verify-only`, `--workers N`, `--store PATH`. The mirror
is resumable and idempotent: re-running skips objects already present at the correct size.

## Development

```bash
git clone https://github.com/twangodev/squawk && cd squawk
uv sync
uv run poe check   # ruff lint + format-check + ty typecheck
uv run poe test    # pytest (fast; network marker is default-off)
```
