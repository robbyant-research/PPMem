# Inference guide

See the [README](../README.md) for installation, model weights, and a complete bundled example. Run the commands below from the repository root.

## Model selection

Weights are downloaded automatically from `Robbyant-Research/PPMem` and `Ruicheng/moge-2-vitl-normal` and cached by Hugging Face. Use `--checkpoint` for a local PPMem directory or another Hub repo ID, and `--moge-checkpoint` for a local MoGe weight file or another Hub repo ID. Use a `./` prefix for relative local paths so that a missing path is not mistaken for a repo ID. With `--init_geometry`, MoGe is not loaded.

Use local paths or a populated Hub cache with `HF_HUB_OFFLINE=1` when offline. Missing local checkpoints report an error rather than silently selecting another model.

## Use your own image

```bash
python scripts/make_trajectory.py \
  --shape dolly --dist 0.5 --frames 121 --out trajectory.npy

bash scripts/run_demo.sh \
  --image input.png \
  --extrinsic trajectory.npy \
  --out runs/demo \
  --prompt "A camera moves slowly forward through the scene, then returns." \
  --seed 42
```

The image is center-cropped to 16:9 and resized to 1024 × 576. MoGe estimates its initial depth and intrinsics. The generated trajectory moves forward and back with a maximum displacement of 0.5 meters in MoGe's estimated scene scale.

The result is `runs/demo/pool/generated_full.mp4`; the log is `runs/demo/run.log`. Use a new or empty output directory; a run that failed before producing output leaves only its log, and the same command can be rerun. To replace an unfinished run, add `--overwrite`. Completed runs are skipped; choose a new directory to run them again.

## Inputs and camera trajectories

**Image.** The input is center-cropped to 16:9, then resized to 1024 × 576.

**Camera trajectory.** `--extrinsic` takes a `.npy` file containing an **`(N, 4, 4)` float32 array of camera-to-world transforms**. Use the OpenCV camera convention: **+X right, +Y down, +Z forward**. Translation uses the same units as the initial depth map, which with MoGe are estimated meters.

Each chunk has 25 frames and overlaps the previous chunk by one frame; the last chunk takes whatever poses are left, so a trajectory of any length is generated to its end. `--max_steps K` selects the **last chunk index**, so `--max_steps 4` produces five chunks and 121 frames. Omit it to run to the end of the trajectory.

The trajectory helper supports the following motions. It takes any frame count from 2 up and a `--dist` value for every shape; use `--dist 0` for `spin`.

| `--shape` | Motion | Control |
| --- | --- | --- |
| `dolly` | Move forward and return with a fixed orientation. | `--dist` sets the maximum displacement. |
| `pendulum` | Repeat a smooth forward-and-backward motion. | `--cycles` sets the round trips; default 2. |
| `spin` | Rotate in place around the vertical axis. | `--turns` sets the turns; default 1. |

Use gradual motion relative to the scene's depth. Increase the frame count to reduce translation or rotation between consecutive poses.

**Depth and intrinsics.** MoGe estimates both from the input image. Pass `--init_geometry /path/to/geometry.npz` to supply them instead — a depth sensor, a different depth model, or a scene whose true scale you know:

```python
import numpy as np
np.savez("geometry.npz",
         depth=depth,          # (H, W) float32, metres — the same units as the trajectory
         fx=fx, fy=fy,         # pixels, at this depth's resolution
         cx=cx, cy=cy)         # principal point, pixels, at this depth's resolution
```

| Key | Contents |
| --- | --- |
| `depth` | The first frame's depth map: `(H, W)` float32 in metres. A frame axis is rejected — only this one frame is ever used. |
| `fx` `fy` `cx` `cy` | Scalars in pixels, describing `depth` at its own `(H, W)`. |
| `units` | Optional, `"m"`. |

Depth and intrinsics travel in one file because neither is meaningful without the other: intrinsics only describe a depth map at the resolution it is stored at. PPMem therefore resizes the two together. A 16:9 depth at any resolution is resized to 1024 × 576, and `fx`, `fy`, `cx`, `cy` are scaled by the same factors. A depth with another aspect ratio is accepted only when it has the input image's exact size; the depth and the image are then center-cropped to 16:9 with the same box, which also shifts `cx` and `cy`. Other combinations are rejected, because which part of the depth the video covers cannot be known. The principal point may be off-center but must lie inside the frame. `np.load` reads `.npz` with `allow_pickle=False`, so this file cannot execute code.

The depth must show the same view as the input image. Storing it at the image's own resolution is the simplest option; a depth you crop to 16:9 yourself must match the image's center crop.

## Run controls

The demo uses [configs/inference.yaml](../configs/inference.yaml) and forwards inference options to the Python CLI. Run `ppmem --help` for the full list.

| Option | Default | Meaning |
| --- | --- | --- |
| `--max_steps` | Derived from the trajectory | Last chunk index to generate, inclusive. |
| `--start_step` | `0` | First chunk index; positive values resume an existing pool. |
| `--seed` | `42` | Base seed; chunk N uses `seed + N`. |
| `--run.num_inference_steps` | `50` | Denoising steps per chunk. |
| `--top_k` | `2` | Source views projected to each target pose; 1 or 2. |
| `--num_context` | `5` | Selected context frames; 0 disables them. |

Override runtime settings with `--section.key value`, for example `--run.num_inference_steps 30`. Available fields are defined in [ppmem/config.py](../ppmem/config.py). Add `--run.deterministic` to enable deterministic algorithm settings.

### Resume a run

Resume from a **completed chunk boundary** with the pool intact. After completing chunks 0 and 1 with `--max_steps 1`, continue the same 121-pose trajectory with:

```bash
bash scripts/run_demo.sh \
  --image input.png --extrinsic trajectory.npy --out runs/demo \
  --start_step 2 --max_steps 4 \
  --prompt "A camera moves slowly forward through the scene, then returns." \
  --seed 42
```

Pass the same checkpoint, configuration, trajectory, prompt, and generation settings on each invocation.

### Python CLI

Use `ppmem` or `python -m ppmem.cli` to invoke the pipeline directly:

```bash
ppmem \
  --config configs/inference.yaml \
  --image input.png --extrinsic trajectory.npy --output_root runs/direct \
  --prompt "A camera moves slowly forward through the scene, then returns." \
  --seed 42
```

`--overwrite` and automatic `run.log` capture are provided by `scripts/run_demo.sh`.
