import functools
import math
import typing

import einops
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class DiTConfig(typing.NamedTuple):
    input_dimension: int
    hidden_dimension: int
    head_dimension: int
    condition_dimension: int
    layers: int
    patch_size: int


class Buffer(nnx.Variable): ...


class TimeEmbedding(nnx.Module):
    def __init__(self, condition_dimension: int, frequency_dimension: int = 1024, max_frequency: float = 100.0, *, rngs: nnx.Rngs) -> None:
        self.frequency = Buffer(jnp.exp(jnp.linspace(math.log(1.0), math.log(max_frequency), frequency_dimension // 2)))
        self.linear = nnx.Linear(frequency_dimension, condition_dimension, use_bias=True, rngs=rngs)

    def __call__(self, time: jax.Array) -> jax.Array:
        x = time * (math.pi / 2) * self.frequency
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)
        x = self.linear(x)

        return x


class PatchEmbedding(nnx.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, patch_size: int, *, rngs: nnx.Rngs) -> None:
        self.patch_size = patch_size
        self.linear = nnx.Linear(input_dimension * patch_size**2, hidden_dimension, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, int, int]:
        height, width = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = einops.rearrange(x, "b c (h p) (w q) -> b (h w) (c p q)", p=self.patch_size, q=self.patch_size)
        x = self.linear(x)

        return x, height, width


class PatchUnembedding(nnx.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, patch_size: int, *, rngs: nnx.Rngs) -> None:
        self.patch_size = patch_size
        self.linear = nnx.Linear(hidden_dimension, input_dimension * patch_size**2, use_bias=False, kernel_init=nnx.initializers.zeros, rngs=rngs)

    def __call__(self, x: jax.Array, height: int, width: int) -> jax.Array:
        x = self.linear(x)
        x = einops.rearrange(x, "b (h w) (c p q) -> b c (h p) (w q)", h=height, w=width, p=self.patch_size, q=self.patch_size)

        return x


class AdaNorm(nnx.Module):
    def __init__(self, hidden_dimension: int, condition_dimension: int, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(condition_dimension, hidden_dimension * 2, use_bias=True, kernel_init=nnx.initializers.zeros, rngs=rngs)
        self.norm = nnx.RMSNorm(hidden_dimension, use_scale=False, rngs=rngs)

    def __call__(self, x: jax.Array, condition: jax.Array) -> jax.Array:
        shift, scale = jnp.split(self.linear(condition), 2, axis=-1)
        x = self.norm(x) * (scale + 1) + shift

        return x


class MLP(nnx.Module):
    def __init__(self, hidden_dimension: int, condition_dimension: int, factor: int = 4, *, rngs: nnx.Rngs) -> None:
        self.linear_1 = nnx.Linear(hidden_dimension, hidden_dimension * factor, use_bias=False, rngs=rngs)
        self.linear_2 = nnx.Linear(hidden_dimension, hidden_dimension * factor, use_bias=False, rngs=rngs)
        self.linear_3 = nnx.Linear(hidden_dimension * factor, hidden_dimension, use_bias=False, kernel_init=nnx.initializers.zeros, rngs=rngs)
        self.ada_norm = AdaNorm(hidden_dimension, condition_dimension, rngs=rngs)

    def __call__(self, x: jax.Array, condition: jax.Array) -> jax.Array:
        x = self.ada_norm(x, condition)
        x = self.linear_1(x) * nnx.silu(self.linear_2(x))
        x = self.linear_3(x)

        return x


class Attention(nnx.Module):
    def __init__(self, hidden_dimension: int, condition_dimension: int, head_dimension: int, *, rngs: nnx.Rngs) -> None:
        self.heads = hidden_dimension // head_dimension
        self.linear_1 = nnx.Linear(hidden_dimension, hidden_dimension * 3, use_bias=False, rngs=rngs)
        self.linear_2 = nnx.Linear(hidden_dimension, hidden_dimension, use_bias=False, kernel_init=nnx.initializers.zeros, rngs=rngs)
        self.q_norm = nnx.RMSNorm(head_dimension, rngs=rngs)
        self.k_norm = nnx.RMSNorm(head_dimension, rngs=rngs)
        self.ada_norm = AdaNorm(hidden_dimension, condition_dimension, rngs=rngs)

    def __call__(self, x: jax.Array, condition: jax.Array, rope: np.ndarray) -> jax.Array:
        x = self.ada_norm(x, condition)
        q, k, v = einops.rearrange(self.linear_1(x), "b l (n h d) -> n b l h d", n=3, h=self.heads)
        q = apply_rope(self.q_norm(q), rope)
        k = apply_rope(self.k_norm(k), rope)
        x = nnx.dot_product_attention(q, k, v)
        x = self.linear_2(einops.rearrange(x, "b l h d -> b l (h d)"))

        return x


class DiTBlock(nnx.Module):
    def __init__(self, config: DiTConfig, *, rngs: nnx.Rngs) -> None:
        self.attention = Attention(config.hidden_dimension, config.condition_dimension, config.head_dimension, rngs=rngs)
        self.mlp = MLP(config.hidden_dimension, config.condition_dimension, rngs=rngs)

    def __call__(self, x: jax.Array, condition: jax.Array, rope: np.ndarray) -> jax.Array:
        x = x + self.attention(x, condition, rope)
        x = x + self.mlp(x, condition)

        return x


class DiT(nnx.Module):
    def __init__(self, config: DiTConfig, rngs: nnx.Rngs) -> None:
        self.config = config
        self.time_embedding = TimeEmbedding(config.condition_dimension, rngs=rngs)
        self.patch_embedding = PatchEmbedding(config.input_dimension, config.hidden_dimension, config.patch_size, rngs=rngs)
        self.patch_unembedding = PatchUnembedding(config.input_dimension, config.hidden_dimension, config.patch_size, rngs=rngs)
        self.blocks = nnx.List([DiTBlock(config, rngs=rngs) for _ in range(config.layers)])
        self.ada_norm = AdaNorm(config.hidden_dimension, config.condition_dimension, rngs=rngs)

    def __call__(self, x: jax.Array, time: jax.Array) -> jax.Array:
        x, height, width = self.patch_embedding(x)
        time = self.time_embedding(time[:, None, None])
        rope = create_rope_2d(self.config.head_dimension, height, width)

        for block in self.blocks:
            x = block(x, time, rope)

        x = self.patch_unembedding(self.ada_norm(x, time), height, width)

        return x


def create_rope_1d(head_dimension: int, sequence_length: int) -> np.ndarray:
    frequency = np.exp(np.linspace(math.log(1.0), math.log(sequence_length / 4), head_dimension // 2))
    x = np.linspace(-math.pi / 2, math.pi / 2, sequence_length)
    x = x[..., None] * frequency
    x = np.stack([np.cos(x), np.sin(x)], axis=0).repeat(2, axis=-1)

    return x


@functools.lru_cache(maxsize=None)
def create_rope_2d(head_dimension: int, height: int, width: int) -> np.ndarray:
    rope_height = np.repeat(create_rope_1d(head_dimension // 2, height)[:, :, None], width, axis=2)
    rope_width = np.repeat(create_rope_1d(head_dimension // 2, width)[:, None, :], height, axis=1)
    x = np.concatenate([rope_height, rope_width], axis=-1).reshape(2, -1, 1, head_dimension)

    return x


def apply_rope(x: jax.Array, rope: np.ndarray) -> jax.Array:
    cos, sine = rope
    x_right = jnp.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)

    return x * cos + x_right * sine
