# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: softjax
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Differentiable Rasterizer
# Code based on:
# - https://github.com/SebLague/Software-Rasterizer
# - https://github.com/ShichenLiu/SoftRas

# %% [markdown]
# ## Helper functions
#
#

# %%
# Show image
import jax.numpy as jnp
import matplotlib.pyplot as plt


def render(img, title=None):
    """Display a 2D (H, W) or 3D (H, W, 3) jax.numpy image."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(jnp.asarray(img), cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    plt.show()

# Example RGB image: three coloured gaussian blobs on a 64x64 grid.
H, W = 64, 64
colourbuffer = jnp.zeros((H, W, 3))
# yy, xx = jnp.mgrid[0:H, 0:W]


# def _blob(cx, cy, sigma):
#     return jnp.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))


# r = _blob(W * 0.35, H * 0.40, W / 8)
# g = _blob(W * 0.65, H * 0.40, W / 8)
# b = _blob(W * 0.50, H * 0.65, W / 8)
# example_image = jnp.stack([r, g, b], axis=-1)
render(colourbuffer, title="example image")


# %%
# Math helpers (JAX ports of Software-Rasterizer/Source/Core/Helpers/Maths.cs).
DEGREES_TO_RADIANS = jnp.pi / 180


def signed_parallelogram_area(a, b, c):
    """2x signed area of triangle abc (positive for clockwise winding)."""
    return (c[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1]) \
         + (c[..., 1] - a[..., 1]) * (a[..., 0] - b[..., 0])


def point_in_triangle(a, b, c, p):
    """Test whether p lies inside triangle abc.

    Returns (inside, weight_a, weight_b, weight_c) where the weights are
    barycentric coordinates. Non-clockwise triangles are treated as back-faces.
    """
    area_abp = signed_parallelogram_area(a, b, p)
    area_bcp = signed_parallelogram_area(b, c, p)
    area_cap = signed_parallelogram_area(c, a, p)
    total = area_abp + area_bcp + area_cap
    inv = 1.0 / total
    weight_a = area_bcp * inv
    weight_b = area_cap * inv
    weight_c = area_abp * inv
    inside = (area_abp >= 0) & (area_bcp >= 0) & (area_cap >= 0) & (total > 0)
    return inside, weight_a, weight_b, weight_c


def clamp01(x):
    return jnp.clip(x, 0.0, 1.0)


def clamp(x, lo, hi):
    return jnp.clip(x, lo, hi)


def lerp(a, b, t):
    return a + (b - a) * clamp01(t)


def remap01(value, lo, hi):
    return clamp01((value - lo) / (hi - lo))


def round_to_int(x):
    return jnp.round(x).astype(jnp.int32)


def to_radians(deg):
    return deg * DEGREES_TO_RADIANS


# %% [markdown]
# ## 2D rasterizer example
# After the triangle vertices are projected onto the camera plane to yield 2D points $a_i$, $b_i$, $c_i$, we have to determine if each pixel $p_i$ lies inside these vertices.  

# %%

# Triangle in 2D formed by three vertices (x, y) and a color (r, g, b).

