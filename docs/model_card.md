---
tags:
  - mnist
  - flow-matching
  - diffusion
  - numerical-analysis
  - pytorch
  - educational
---
# MNIST conditional Flow and DDPM — BS6221

Small class-conditional U-Nets for studying noise-to-image generation and numerical
ODE sampling. These are educational image models, not biological/scRNA-seq models.

## Files and architecture

- `flow_ema.pt`: Flow velocity predictor, 15,000 optimizer updates, EMA inference weights.
- `diffusion_ema.pt`: DDPM noise predictor, 20,000 updates, EMA inference weights; 100-step cosine schedule.
- `digit_classifier.pt`: auxiliary MNIST classifier; its labels are a proxy, not comprehensive generation quality.
- `model_definitions.py`: frozen PyTorch definitions matching these weights.
- `manifest.json`: model provenance and SHA-256 checksums.
- `config.json`: image shape, normalization, architecture and training budgets.

Each generator has 305,289 parameters, 24 base channels, a time embedding and
10-class digit embedding. Inputs have shape `[B,1,28,28]`, real-image normalization
is `pixel / 127.5 - 1`. Labels are integer digits 0–9.

## Data and selection

MNIST official training images were split into 55,000 training and 5,000 validation
images using the saved seed-42 split. The official 10,000-image test set was not
used for generator training or checkpoint selection. The continuation selected
EMA checkpoints by held-out validation MSE on 1,024 images with two fixed
noise/time draws. See the manifest for source hashes and selection losses.

Flow predicts the conditional velocity `image - noise` by MSE. DDPM predicts added
noise by MSE. These losses have different targets and cannot be compared directly.
The training budgets differ; this is not an equal-training-budget benchmark.

## Loading

These are custom PyTorch checkpoints, not a Transformers/Diffusers pipeline.
Download a pinned commit and verify the manifest hashes, then instantiate
`ConditionalTinyUNet(24)` and load the checkpoint's `state_dict` with
`torch.load(path, map_location='cpu', weights_only=True)`.
Generation APIs and the interactive notebook are provided in the accompanying
BS6221 GitHub project; its URL will be added after publication.

## Intended use and limitations

For coursework, reproducible solver comparisons and interactive MNIST teaching.
Some generated samples may not match the requested digit. No claim of
state-of-the-art image quality, causal/biological interpretation or medical use.
ODE references are numerical approximations, and refinement differences are
not rigorous error bounds. CUDA/Windows/Linux compatibility has not been verified
on physical machines; timing comparisons are device-specific.

## References

- MNIST: https://yann.lecun.org/exdb/mnist/
- Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2006.11239

No open-source license has been selected yet. Review licensing before wider reuse.
