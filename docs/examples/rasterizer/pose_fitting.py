"""Rigid pose fitting on the differentiable rasterizer.

A "start test" for whether the gradients of ``rasterizer_functions.render``
are usable for gradient-based optimization. We render a sphere at an unknown
rotation with *hard* rasterization (the target image), then fit a rotation
matrix so that the *smooth* (differentiable) render of the same sphere
matches the target under a Euclidean (L2) image loss.

Rotation parameterization
-------------------------
The optimization variable is a free 3x3 matrix ``M`` (9 unconstrained
numbers). Before use it is projected onto SO(3) via SVD-based symmetric
orthogonalization (Levinson et al. 2020, "An Analysis of SVD for Deep
Rotation Estimation"). Gradients flow through the SVD, so plain
``jax.value_and_grad`` over ``M`` is all that is needed -- ``M`` is a single
``jax.Array`` with no static leaves, so ``eqx.filter_grad`` buys nothing.

Training loop follows the equinox MNIST example: a jitted ``make_step`` that
computes ``value_and_grad``, then ``optax`` adam ``update`` / ``apply_updates``.

Output
------
Writes ``pose_fitting.png``: the L2 loss curve, the geodesic rotation-error
curve, and a target / initial / estimated render comparison.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

import rasterizer_functions as rf


# --- Rotation parameterization -------------------------------------------

def svd(m):
    """Map a 3x3 matrix onto SO(3) via symmetric orthogonalization.

    Source: Google Research, ``special_orthogonalization/utils.py``
    https://github.com/google-research/google-research/blob/193eb9d/special_orthogonalization/utils.py#L93-L115
    """
    U, _, Vh = jnp.linalg.svd(m, full_matrices=False)
    det = jnp.linalg.det(jnp.matmul(U, Vh))
    return jnp.matmul(jnp.c_[U[:, :-1], U[:, -1] * det], Vh)


def geodesic_angle(r_a, r_b):
    """Geodesic distance (radians) between two rotation matrices."""
    cos = (jnp.trace(r_a.T @ r_b) - 1.0) / 2.0
    return jnp.arccos(jnp.clip(cos, -1.0, 1.0))


def y_angle(r):
    """Y-rotation angle (radians) of a rotation matrix.

    Exact for ``rf._rotation_y`` outputs; an approximation otherwise.
    """
    return jnp.arctan2(r[0, 2], r[0, 0])


# --- Pose fitting ---------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # Optimization at a modest resolution keeps ~400 jitted iterations fast.
    H, W = 128, 128
    angle_true_deg = 180   # unknown rotation we try to recover
    learning_rate = 1e-2
    n_steps = 400

    # --- Geometry ---------------------------------------------------------
    # The sphere's loss landscape descends monotonically from 0 deg to the
    # target with ~32% relative depth at the default render softness, so no
    # softness annealing is needed -- plain gradient descent converges. (A
    # faceted object such as the cube has a depth-ordering loss ridge near
    # ~15 deg that traps the 0-deg init; a sphere has no such ridge.)
    sphere_v, sphere_n, sphere_t = rf._load_obj(os.path.join(here, "sphere.obj"))

    camera = rf.Camera(
        fov=jnp.asarray(jnp.pi / 3),
        transform=rf.Transform(
            position=jnp.array([0.0, 0.0, -3.0]),
            rotation=jnp.eye(3),
            scale=0.7,
        ),
    )

    def empty_target():
        return rf.RenderTarget(
            color_buffer=jnp.zeros((H, W, 3)),
            depth_buffer=jnp.full((H, W), jnp.inf),
        )

    def make_scene(rotation):
        """Scene with the sphere at ``rotation`` over a flat white background."""
        sphere = rf.Model(
            vertices=sphere_v,
            tex_coords=sphere_t,
            normals=sphere_n,
            transform=rf.Transform(
                position=jnp.zeros(3),
                rotation=rotation,
                scale=jnp.asarray(1.5, dtype=jnp.float32),
            ),
            shader=rf._normal_shader,
        )
        return rf.SceneData(camera=camera, models=[sphere])

    def render_color(rotation, mode):
        return rf.render(empty_target(), make_scene(rotation),
                         mode=mode).color_buffer

    # --- Target: sphere at an unknown rotation, hard rasterization --------
    R_true = rf._rotation_y(jnp.deg2rad(angle_true_deg))
    target_image = render_color(R_true, mode="hard")

    # --- Loss: L2 between smooth render and the hard target ---------------
    def loss_fn(M):
        pred = render_color(svd(M), "smooth")
        return jnp.mean((pred - target_image) ** 2)

    # --- Optimizer (equinox-style jitted step) ----------------------------
    optimizer = optax.adam(learning_rate)
    M = jnp.diag(jnp.array([1.3, 1.1, 0.9]))  # Init values
    opt_state = optimizer.init(M)

    @jax.jit
    def make_step(M, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(M)
        updates, opt_state = optimizer.update(grads, opt_state, M)
        M = optax.apply_updates(M, updates)
        return M, opt_state, loss

    # --- Training loop ----------------------------------------------------
    losses, rot_errors = [], []
    print(f"Fitting rotation: target = {angle_true_deg:.1f} deg about Y, "
          f"init = 0 deg, lr = {learning_rate}, steps = {n_steps}")
    for step in range(n_steps):
        print(f"Step {step:4d}/{n_steps}...", end="")
        M, opt_state, loss = make_step(M, opt_state)
        err_deg = float(jnp.rad2deg(geodesic_angle(svd(M), R_true)))
        losses.append(float(loss))
        rot_errors.append(err_deg)
        if step % 20 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}   loss {float(loss):.6e}   "
                  f"rot_err {err_deg:7.3f} deg")

    R_est = svd(M)
    est_deg = float(jnp.rad2deg(y_angle(R_est)))
    print(f"Done. estimated Y-angle = {est_deg:.2f} deg "
          f"(target {angle_true_deg:.1f} deg), "
          f"final rotation error = {rot_errors[-1]:.3f} deg")

    # --- Plots ------------------------------------------------------------
    init_image = render_color(jnp.eye(3), mode="hard")
    est_image = render_color(R_est, mode="hard")

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Rigid pose fitting via the differentiable rasterizer")
    gs = fig.add_gridspec(2, 3)

    panels = [
        (f"target ({angle_true_deg:.0f} deg, hard)", target_image),
        ("initial (0 deg, hard)", init_image),
        (f"estimated ({est_deg:.1f} deg, hard)", est_image),
    ]
    for col, (title, img) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(jnp.clip(img, 0.0, 1.0))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_loss = fig.add_subplot(gs[1, :2])
    ax_loss.semilogy(losses, color="C0")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("L2 image loss")
    ax_loss.set_title("Euclidean image loss", fontsize=10)
    ax_loss.grid(True, alpha=0.3)

    ax_err = fig.add_subplot(gs[1, 2])
    ax_err.plot(rot_errors, color="C1")
    ax_err.set_xlabel("iteration")
    ax_err.set_ylabel("geodesic error (deg)")
    ax_err.set_title("rotation error", fontsize=10)
    ax_err.grid(True, alpha=0.3)

    out_path = os.path.join(here, "pose_fitting.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
