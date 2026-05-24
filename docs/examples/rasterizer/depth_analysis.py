"""Visualise per-pixel depth produced by `rasterize_triangle` for one triangle.

Builds a single-triangle `RasterizedModel` with hand-chosen screen
positions and vertex depths, runs `rasterize_triangle` over an H x W
pixel grid in both `hard` and `smooth` modes, and plots the resulting
`depth` field (and `inside` for context). The expected ground-truth
linear-barycentric depth is plotted alongside so the centroid-collapse
behaviour of the current clipped-area weights is easy to spot — over
the interior of the triangle the rendered depth is constant (the
centroid of the three vertex depths) instead of the linear ramp shown
in the ground-truth column.
"""

import os
import matplotlib.pyplot as plt
import jax.numpy as jnp

import rasterizer_functions as rf


def make_single_triangle(screen_pos, vertex_depths):
    """Wrap (3, 2) screen positions + (3,) depths in a `RasterizedModel`."""
    return rf.RasterizedModel(
        depth=jnp.asarray(vertex_depths, dtype=jnp.float32),       # (3,)
        screen_pos=jnp.asarray(screen_pos, dtype=jnp.float32),     # (3, 2)
        tex_coords=jnp.zeros((3, 2), dtype=jnp.float32),
        normals=jnp.zeros((3, 3), dtype=jnp.float32),
    )


def ground_truth_depth(a, b, c, depths, h, w):
    """Linear barycentric depth over the full grid (signed weights, no clip)."""
    p = rf._pixel_grid(h, w)                            # (H, W, 2)
    area_abp = rf.signed_parallelogram_area(a, b, p)
    area_bcp = rf.signed_parallelogram_area(b, c, p)
    area_cap = rf.signed_parallelogram_area(c, a, p)
    area_total = area_abp + area_bcp + area_cap
    wa = rf.safe_division(area_bcp, area_total)
    wb = rf.safe_division(area_cap, area_total)
    wc = rf.safe_division(area_abp, area_total)
    return wa * depths[0] + wb * depths[1] + wc * depths[2]


def _dummy_shader(pixel_xy, tex_coord, normal, depth):
    return jnp.zeros((*pixel_xy.shape[:-1], 3), dtype=jnp.float32)


def main():
    H, W = 256, 256
    
    # CW winding order
    # screen_pos = 0.5 * jnp.array([
    #     [ 50.0,  60.0],   # a
    #     [200.0,  80.0],   # b
    #     [120.0, 210.0],   # c
    # ])
    
    # CCW winding order
    screen_pos = 0.5 * jnp.array([
        [120.0, 210.0],   # c
        [200.0,  80.0],   # b
        [ 50.0,  60.0],   # a
    ])
    # Make vertex depths *clearly different* so the interpolation pattern
    # is visible. A correct rasterizer should show a smooth gradient
    # across the triangle interior between these values.
    vertex_depths = jnp.array([1.0, 2.0, 3.0])

    triangle = make_single_triangle(screen_pos, vertex_depths)

    softness = 1e-3
    color_h, depth_h, inside_h = rf.rasterize_triangle(
        triangle, H, W, _dummy_shader, mode="hard", softness=softness,
    )
    color_s, depth_s, inside_s = rf.rasterize_triangle(
        triangle, H, W, _dummy_shader, mode="smooth", softness=softness,
    )

    a, b, c = screen_pos[0], screen_pos[1], screen_pos[2]
    depth_gt = ground_truth_depth(a, b, c, vertex_depths, H, W)

    # Common depth colour scale across all three depth panels so they
    # are directly comparable.
    d_lo = float(min(jnp.min(depth_h), jnp.min(depth_s)))#, jnp.min(depth_gt)))
    d_hi = float(max(jnp.max(depth_h), jnp.max(depth_s)))#, jnp.max(depth_gt)))

    fig, axes = plt.subplots(2, 3, figsize=(3 * 3.4, 2 * 3.4),
                             constrained_layout=True)
    fig.suptitle(
        f"rasterize_triangle depth — vertex depths {tuple(float(v) for v in vertex_depths)}, "
        f"softness={softness}",
        fontsize=11,
    )

    panels = [
        ("hard:   depth (rendered)",  depth_h,    "viridis", (d_lo, d_hi)),
        ("smooth: depth (rendered)",  depth_s,    "viridis", (d_lo, d_hi)),
        ("ground truth: bary depth",  depth_gt,   "viridis", (d_lo, d_hi)),
        ("hard:   inside",            inside_h,   "gray",    (0.0, 1.0)),
        ("smooth: inside",            inside_s,   "gray",    (0.0, 1.0)),
        ("|rendered_smooth - gt|",    jnp.abs(depth_s - depth_gt),
                                                  "magma",  (0.0, float(jnp.max(jnp.abs(depth_s - depth_gt))))),
    ]

    for ax, (title, field, cmap, (vmin, vmax)) in zip(axes.ravel(), panels):
        im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        # Triangle outline (note: imshow with origin="upper" uses pixel
        # coordinates directly, so x = column, y = row).
        xs = [float(a[0]), float(b[0]), float(c[0]), float(a[0])]
        ys = [float(a[1]), float(b[1]), float(c[1]), float(a[1])]
        ax.plot(xs, ys, color="white", linewidth=1.2)
        for v, lab in zip([a, b, c], "abc"):
            ax.scatter([float(v[0])], [float(v[1])], color="white", s=18)
            ax.annotate(f"{lab} (d={float(vertex_depths['abc'.index(lab)]):.1f})",
                        (float(v[0]), float(v[1])), color="white",
                        xytext=(4, 4), textcoords="offset points", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "depth_analysis.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")

    # Quick numerical readout at the centroid pixel.
    cx = int((float(a[0]) + float(b[0]) + float(c[0])) / 3)
    cy = int((float(a[1]) + float(b[1]) + float(c[1])) / 3)
    print(f"At centroid pixel ({cx}, {cy}):")
    print(f"  hard   rendered depth = {float(depth_h[cy, cx]):.4f}")
    print(f"  smooth rendered depth = {float(depth_s[cy, cx]):.4f}")
    print(f"  ground-truth depth    = {float(depth_gt[cy, cx]):.4f}  "
          f"(centroid of vertex depths = {float(jnp.mean(vertex_depths)):.4f})")


if __name__ == "__main__":
    main()
