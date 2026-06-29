# long chen / uuirs

ML systems engineer focused on JAX, model inference, and on-device AI.

I work on practical inference systems across JAX, PyTorch, CoreML, ONNX, WebGPU, and Swift. My current focus is lightweight video segmentation/tracking, edge deployment, and long-sequence streaming models.

## Selected open-source contributions

### JAX

- [jax-ml/jax #27292](https://github.com/jax-ml/jax/pull/27292) — Fix wrong unpacking in the decode attention Pallas kernel  
  Merged. Fixed an output-structure mismatch in the decode attention Pallas kernel that could cause shape mismatches and incorrect assertions when `batch_size > 1`.

### Equinox

- [patrick-kidger/equinox #235](https://github.com/patrick-kidger/equinox/pull/235) — Support buffer donation in `filter_jit`  
  Merged. Added support for controlling JAX buffer donation through `donate_default`, `donate_args`, `donate_kwargs`, and `donate_fn`.

- [patrick-kidger/equinox #233](https://github.com/patrick-kidger/equinox/pull/233) — Fix `tree_inference`  
  Merged. Fixed repeated node collection in nested PyTrees, reducing unnecessary traversal work in deeply nested module structures.

- [patrick-kidger/equinox #190](https://github.com/patrick-kidger/equinox/pull/190) — Fix attention mask handling and add tests  
  Merged. Fixed mask axis handling in attention and added regression tests.

## Current work

### Edge video tracking / segmentation

Building lightweight video object tracking and segmentation pipelines for mobile and browser deployment.

Areas involved:

- EdgeTAM / SAM2-style video tracking
- CoreML model splitting and quantization
- Swift native inference
- ONNX / WebGPU deployment
- Mask generation for interactive visual effects

### SeqT-JAX

Long-sequence streaming Transformer system for time-series inference.

Highlights:

- Long-context sequence modeling
- Streaming inference with cached state
- Ring-buffer based state management
- Practical latency reduction for online inference workloads

## Technical focus

- JAX / PyTorch / Equinox
- Pallas kernels and inference correctness
- CoreML / ONNX / WebGPU
- Swift / iOS native inference
- Long-sequence and streaming model systems
- Edge AI product prototyping

## Selected repositories

- [JAX fork](https://github.com/uuirs/jax)
- [Equinox fork](https://github.com/uuirs/equinox)
- [diffusers fork](https://github.com/uuirs/diffusers)
- [mica-flow](https://github.com/uuirs/mica-flow)
- [sam](https://github.com/uuirs/sam)

---

I prefer building small, verifiable systems: model export, numerical comparison, runtime integration, and deployment on constrained devices.
