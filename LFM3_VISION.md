# LFM3 still-image understanding

```sh
cargo run -r -- --model LiquidAI/LFM3-VL-2B-GGUF:BF16 \
  --image photo.png --prompt 'Describe this image.' -n 128
cargo run -r -- --model LiquidAI/LFM3-VL-2B-GGUF:BF16 \
  --image first.jpg --image second.jpg \
  --prompt 'Compare these images.' --image-max-pixels 262144 -n 128
```

The standard Hugging Face cache holds the language checkpoint and the automatically
selected `mmproj-LFM3-VL-2B-BF16.gguf` companion. `--mmproj-model` selects another
compatible projector repository. PNG and JPEG input and RGB conversion reuse the
existing image path. Image input is available through the CLI and the library's
`VisionModel` / `PreparedMultimodalPrompt` / session APIs.

## CUDA

Set `CATENA_GPU_DIALECT=cuda` to use CUDA for both the vision encoder/projector
and the language model (sm_80+). The language path shares the portable LFM3 text
kernels, including accelerated image-prompt prefill. The vision encoder keeps
its existing F32 kernels, which already support both dialects; preprocessing and
image token insertion are unchanged. HIP remains the default. CUDA execution
and performance remain unverified on hardware.

The complete vision and image-input language graphs can be compiled without a
GPU using the cached checkpoint paths:

```sh
LFM3_VL_PROJECTOR=/path/to/mmproj-LFM3-VL-2B-BF16.gguf CATENA_GPU_DIALECT=cuda \
  cargo test -r --lib vision::kernels::tests::lfm3_graph_compiles -- --ignored
LFM3_VL_GGUF=/path/to/LFM3-VL-2B-BF16.gguf CATENA_GPU_DIALECT=cuda \
  cargo test -r --lib lfm3::tests::multimodal_graph_compiles -- --ignored
```

Set `CATENA_QWEN3_CPP` to a `.cu` path to export either compiled CUDA module for
offline nvcc checks. Both complete CUDA graphs passed nvcc 12.4 PTX, cubin, and
shared-library compilation; the language graph contains native MMA/HMMA. The
existing VL GPU tests also honor `CATENA_GPU_DIALECT`. The port passes the HIP
red-image prefill/decode/resume check ("The image is red.") and retains the
reference embedding errors reported below.

## Model path

The language model reuses LFM3 text weights, tokenizer, convolution history, full
and sliding attention, WMMA prefill, decode kernels, and checkpoint handling.
A separate embedding-input entrypoint runs the same decoder on the entire image
prompt as one batch. A GPU copy inserts projected image rows into token embeddings
and narrows them to BF16. Text-only LFM3 source generation is unchanged.

The checkpoint's chat template supplies BOS, image boundaries, and the assistant
prefix with thinking disabled, consistent with the existing image CLI. Each
`<|image|>` marker expands to the projected image's token count. Positions remain
linear across image and text rows; this model does not use language-side MRoPE.
Continuation and checkpoint resume retain that position convention.

The 189M vision companion contains a 14-layer encoder with eight gated spatial
convolution blocks and six bidirectional attention blocks. It uses RMS
normalization, grouped query attention, 2D axial RoPE, and SwiGLU. The final
LayerNorm and 2×2 patch merge feed a GELU projector into the 2048-channel language
space. K/V head expansion allows reuse of the existing vision attention helper.
Only RMS, spatial convolution/gating, SwiGLU, patch merge, head expansion, and
embedding insertion need additional Hex kernels. No catena-lang changes are used.

Image decoding, smart resize, normalization, graph composition, linear algebra,
attention, RoPE, GELU, buffer ownership and release reuse existing helpers. LFM
patches use raster order through the encoder and switch to merge order only at
the projector. Torchvision-compatible antialiased bicubic/bilinear coefficients
are separate from the unchanged Pillow-compatible Qwen resize path. The default
maximum is capped at 4096 output image tokens, matching the GGUF reference runtime;
`--image-max-pixels` can override the pixel budget.

Despite its BF16 name, the published projector stores both patchifier slices in
F16. They are widened exactly once during load. BF16 matrices are widened once on
the GPU; native F32 tensors remain mapped. The vision graph uses existing F32
matrix kernels, so vision encoding is a future performance-tuning opportunity.
The language decoder retains its existing BF16/WMMA paths.

## Validation

Release library/server tests include the existing Qwen image preprocessing
fixtures and half-precision conversion cases. Focused GPU checks:

```sh
cargo test -r --test lfm3_vl -- --ignored --nocapture
```

The image encoder was compared with llama.cpp on deterministic 256×256, 320×256,
and resized 257×193 RGB inputs. The reference used the same checkpoint weights
exactly widened to F32 to match the existing vision compute path. Relative
embedding RMS differences were 0.018%, 0.021%, and 0.038%; resized pixels matched
exactly. Native BF16 llama.cpp execution differed by about 1% on the square input
because it also rounds matrix inputs. These are focused numerical checks, not a
model-quality evaluation.

The integration check recognizes a red image after batched prefill and checkpoint
resume. The CLI also correctly identified ordered red/blue image inputs. On the
validation machine, 256×256 vision encoding took about 4.5 seconds per image;
the two-image, 154-token language prefill took about 158 ms. Optional `LFM3_VL_REFERENCE` points to the prefix of little-endian F32
reference buffers named `PREFIX-WxH.f32` and normalized resized RGB buffers named
`PREFIX-WxH-pixels.f32`. The latter use interleaved RGB order. All hardware checks
in this pass used HIP/Strix Halo.

Generate the new Hex assets with `python3 tools/generate-lfm3-vision.py`.
