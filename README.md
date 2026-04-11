# dit

An implementation of Diffusion Transformer (DiT) from [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) (Peebles et al., 2022).

## Installation

```sh
uv add git+https://github.com/flawley/dit
```

## Usage

```python
import jax
from flax import nnx

import dit


rngs = nnx.Rngs(0)

model = dit.DiT(
    config=dit.DiTConfig(
        input_dimension=...,
        hidden_dimension=256,
        head_dimension=64,
        condition_dimension=256,
        layers=4,
        patch_size=2,
    ),
    rngs=rngs,
)

x = jax.numpy.array(...) # Shape: (B, C, H, W).
time = rngs.uniform(x.shape[0])  # Shape: (B,).
prediction = model(x, time)  # Shape: (B, C, H, W).
```
