# semanticwct

Photorealistic style transfer backend. The content and style images go through a Haar wavelet
autoencoder, their colour statistics get matched with a whitening and colouring transform (WCT),
and the result is decoded and smoothed back over the content with a guided filter.

WCT can run globally or per semantic segment (Grounding-DINO boxes, SAM2 masks). Everything is
CPU only, with the models compiled through `torch.compile(mode="max-autotune")`.

What happens to one request:

```
bytes
  -> decode, resize to 512 (aspect kept, sides rounded to multiples of 16)
  -> optional: Grounding-DINO + SAM2, segments matched between content and style
  -> WaveEncoder, 4 levels
  -> WCT on the pool1/pool2/pool3 skips and on all 4 decoder levels
  -> WaveDecoder
  -> resize back to the original size
  -> guided filter, guided by the original content
  -> PNG bytes
```

Style statistics are cached in `registers` inside `wct.py`, so a style image is only analysed once
per request instead of once per WCT call.

## Files

| File | What it does |
|---|---|
| `pipeline.py` | `TransferPipeline`. Wires everything together: preprocessing, WCT, guided filter. Also has a small batch runner, `python pipeline.py` |
| `wavelet_model.py` | `WaveEncoder` and `WaveDecoder`, the Haar wavelet pooling/unpooling autoencoder. Two unpooling modes, `sum` and `cat5` |
| `wct.py` | Whitening and colouring transform: covariance, SVD, rank truncation, the style statistic cache, per-segment path |
| `segmentation.py` | Grounding-DINO detection, SAM2 masks, merging per class, matching segments between the content and style image, IoU resolution when segments overlap |
| `smoothing.py` | Guided filter helper |
| `server.py` | FastAPI service with Supabase JWT auth and background jobs (`/transfer`, `/jobs/{id}`) |
| `ckpts/` | Trained wave encoder/decoder weights, `sum` and `cat5`, level 4 |
| `profiler.py`, `torch_profiler.py`, `trace_analyzer.py` | Profiling: memory and time per stage, `torch.profiler` traces, and a reader for the Chrome traces those produce |
| `benchmark_segmentation.py` | Times the SAM2 and Grounding-DINO stages line by line |
| `benchmark_mask_resize.py` | Compares mask resize methods at the VGG feature map sizes |
| `content/`, `styles/` | Empty, so you can drop `content/<i>.jpg` and `styles/<i>.jpg` in for the batch demo |

## Running

```bash
uv sync
python pipeline.py                        # batch demo, writes to outputs/
python -m uvicorn server:app --port 8000  # API, needs SUPABASE_JWT_SECRET
```

Things worth knowing before you start:

- No images are included. The batch demo looks for `content/1.jpg` and `styles/1.jpg`, creates
  `outputs/` itself and only warns about missing pairs. `segmentation.py` writes its debug masks as
  `<n>_<label>.png` into the working directory.
- Dependencies: `torch>=2.7.1`, `transformers`, `sam-2`, `opencv-contrib-python` (only for
  `cv2.ximgproc.guidedFilter`), `fastapi`, `pyjwt`, `celery`, `pillow`.
- `pyproject.toml` and `uv.lock` are untouched from the original project and point at a local SAM2
  checkout at `../../sam2`. If you don't have one, change that source to
  `{ git = "https://github.com/facebookresearch/sam2" }` and re-lock.
- `SUPABASE_JWT_SECRET` has to be exported before starting `server.py`. No secrets are stored in
  this repo.
- Run everything from the repo root. `segmentation.py` and `pipeline.py` resolve `ckpts/`,
  `sam2_checkpoints/` and `configs/sam2.1/` relative to the working directory.
