# Changelog

## 0.1.0 (2026-06-01)


### Features

* 16kHz WAV resample and encode ([98778ed](https://github.com/twangodev/squawk/commit/98778ed2e2fc5f2ed8617d34b6f85f95a63d26aa))
* clip and ADS-B raw readers ([6d07d00](https://github.com/twangodev/squawk/commit/6d07d0054ad24f47aeff2d6b131b87325892c3dc))
* configurable VAD --batch-size (GPU window batching, default 1024) ([0ba88b2](https://github.com/twangodev/squawk/commit/0ba88b24a3730382dcdbe60121740b9912813f4f))
* core models, config, and constants ([8d93957](https://github.com/twangodev/squawk/commit/8d93957057473240491c658d1bffc02d3dd3be34))
* jsonl manifest ledger ([bb2a320](https://github.com/twangodev/squawk/commit/bb2a3204ad547523b0f2b7671abc713225957495))
* pack clips into embedded sharded parquet + pack CLI ([1c49c8c](https://github.com/twangodev/squawk/commit/1c49c8cd9c38c25470fc51507e000b4a730cc1a3))
* parallel window-join → clips parquet + merge CLI ([ebe2d02](https://github.com/twangodev/squawk/commit/ebe2d0280f51d5c8268fb547bb02677e01658503))
* resumable, atomic download engine ([f6abf09](https://github.com/twangodev/squawk/commit/f6abf091d17ac80abb7c9ff255a5c9445cc00d37))
* source protocol, registry, and tartanaviation source ([0620d58](https://github.com/twangodev/squawk/commit/0620d58cb8390a2e645d4949a15b64847e80faab))
* swift REST client with marker pagination ([0f609ff](https://github.com/twangodev/squawk/commit/0f609ff3fdf35539afb8c933924ff8c488bd668a))
* typer cli — prepare, sources, merge ([44c2859](https://github.com/twangodev/squawk/commit/44c28593681f32e1936962e479a87d1f396e4557))
* VAD segment stage — clips into utterances with re-windowed ADS-B ([5348faa](https://github.com/twangodev/squawk/commit/5348faa488f96eb983eb857b60c2714f697f6c98))


### Bug Fixes

* support Deflate64-compressed ADS-B zips (2022 raw tier) ([a114784](https://github.com/twangodev/squawk/commit/a114784f8eb6412c61249582b61df1a479313a94))
* utterances inherit the full clip's ADS-B (no per-window dropping) ([475e37f](https://github.com/twangodev/squawk/commit/475e37f6d069a24e7341bea0cb21733f799d2e76))


### Documentation

* add README with install and usage ([6216ea5](https://github.com/twangodev/squawk/commit/6216ea522c9c673e6812eb6cb83a7422635dbca4))
* document the prepare→merge→pack→segment pipeline ([f042b23](https://github.com/twangodev/squawk/commit/f042b235e88bbb6ff8bf13e62636d7a05c3dd999))
* minimalist README ([51c5e35](https://github.com/twangodev/squawk/commit/51c5e35261efe7ee5e34db60cf19491c176262e4))
