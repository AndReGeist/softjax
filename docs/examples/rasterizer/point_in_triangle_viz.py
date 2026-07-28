"""Visualise `point_in_triangle` over a 2D plane.

For a fixed triangle (a, b, c) and a dense grid of points p, plots the
four scalar fields returned by `point_in_triangle`: `inside`, `w_a`,
`w_b`, `w_c`. Top row is mode="hard", bottom row is mode="smooth"
at the chosen softness. The triangle edges are overlaid in white.
"""

import os
import matplotlib.pyplot as plt
import jax.numpy as jnp

import rasterizer as rf


def sweep(a, b, c, x_range, y_range, n, mode, softness):
    xs = jnp.linspace(x_range[0], x_range[1], n)
    ys = jnp.linspace(y_range[0], y_range[1], n)
    XX, YY = jnp.meshgrid(xs, ys, indexing="xy")
    p = jnp.stack([XX, YY], axis=-1)  # (n, n, 2)
    return rf.point_in_triangle(a, b, c, p, mode=mode, softness=softness)


def main():
    # Pick a triangle that fits comfortably in the [0, 1]^2 viewport.
    # `point_in_triangle` uses `signed_parallelogram_area >= 0`, which
    # accepts a clockwise winding in math (y-up) coordinates. Order the
    # vertices a -> b -> c clockwise so `inside` is non-zero inside.
    a = jnp.array([0.20, 0.25])
    b = jnp.array([0.50, 0.85])
    c = jnp.array([0.80, 0.30])
    #c = jnp.array([0.55, 0.83])

    # Sweep p over a slightly larger square so we can see extrapolation
    # outside the triangle as well.
    x_range = (-0.1, 1.1)
    y_range = (-0.1, 1.1)
    n = 400
    softness = 1e-1

    out_hard   = sweep(a, b, c, x_range, y_range, n, mode="hard",   softness=softness)
    out_smooth = sweep(a, b, c, x_range, y_range, n, mode="smooth", softness=softness)

    field_names = ["inside", "w_a", "w_b", "w_c"]
    rows = [("hard", out_hard), ("soft", out_smooth)]

    fig, axes = plt.subplots(
        2, 4,
        figsize=(4 * 3.0, 2 * 3.0),
        constrained_layout=True,
    )
    fig.suptitle(
        f"point_in_triangle:  a={tuple(float(v) for v in a)}, "
        f"b={tuple(float(v) for v in b)}, c={tuple(float(v) for v in c)}  "
        f"(softness={softness})",
        fontsize=11,
    )

    for row_idx, (mode_label, fields) in enumerate(rows):
        for col_idx, (name, field) in enumerate(zip(field_names, fields)):
            ax = axes[row_idx, col_idx]
            arr = jnp.asarray(field)
            vmin, vmax = float(jnp.min(arr)), float(jnp.max(arr))
            im = ax.imshow(
                arr,
                extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                origin="lower",
                cmap="viridis",
                vmin=vmin, vmax=vmax,
            )
            # Triangle outline.
            xs = [float(a[0]), float(b[0]), float(c[0]), float(a[0])]
            ys = [float(a[1]), float(b[1]), float(c[1]), float(a[1])]
            ax.plot(xs, ys, color="white", linewidth=1.2)
            ax.scatter([float(a[0])], [float(a[1])], color="white", s=18)
            ax.scatter([float(b[0])], [float(b[1])], color="white", s=18)
            ax.scatter([float(c[0])], [float(c[1])], color="white", s=18)
            ax.annotate("a", (float(a[0]), float(a[1])), color="white",
                        xytext=(4, 4), textcoords="offset points")
            ax.annotate("b", (float(b[0]), float(b[1])), color="white",
                        xytext=(4, 4), textcoords="offset points")
            ax.annotate("c", (float(c[0]), float(c[1])), color="white",
                        xytext=(4, 4), textcoords="offset points")

            ax.set_title(f"{mode_label}: {name}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "point_in_triangle_viz.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
