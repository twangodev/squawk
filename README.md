# squawk

Mirror and align paired ATC audio + ADS-B data.

**Sources:** CMU TartanAviation (KAGC, KBTP)

```bash
pip install squawk-atc

squawk prepare <source>              # mirror audio + ADS-B
squawk merge   <source>              # clip-level parquet + ADS-B tracks
squawk pack    <source>              # embed 16 kHz WAV, shard
squawk segment <packed-dir|hf-repo>  # VAD-split into utterances  (needs [vad])
```

## Datasets

[twangodev/tartanaviation-atc-adsb](https://huggingface.co/datasets/twangodev/tartanaviation-atc-adsb)
