"""Rigid pose fitting on the differentiable rasterizer.

A "start test" for whether the gradients of ``rasterizer_functions.render``
are usable for gradient-based optimization. We render a sphere at an unknown
rotation and translation with *hard* rasterization (the target image), then
fit both the rotation and the translation so that the *smooth*
(differentiable) render of the same sphere matches the target under a
Euclidean (L2) image loss.

Pose parameterization
---------------------
The optimization variable is a tuple ``(M, t)``: ``M`` is a free 3x3 matrix
projected onto SO(3) via SVD-based symmetric orthogonalization (Levinson et
al. 2020, "An Analysis of SVD for Deep Rotation Estimation"), and ``t`` is
a free 3-vector used directly as the object translation. Gradients flow
through the SVD, so plain ``jax.value_and_grad`` over ``(M, t)`` is all
that is needed; both leaves are ``jax.Array`` so ``eqx.filter_grad`` buys
nothing.

Training loop follows the equinox MNIST example: a jitted ``make_step`` that
computes ``value_and_grad``, then ``optax`` adam ``update`` / ``apply_updates``.

Output
------
Writes ``pose_fitting.png``: the L2 loss curve, the geodesic rotation-error
curve, the translation-error curve, and a target / initial / estimated
render comparison.
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

    select_run = "sphere"

    if select_run == "dave":
        # Optimization at a modest resolution keeps ~400 jitted iterations fast.
        H, W = 128, 128
        angle_true_deg = 0  # unknown rotation we try to recover
        R_true = rf._rotation_y(jnp.deg2rad(angle_true_deg)) @ rf._rotation_x(jnp.deg2rad(180))
        position_true = jnp.array([0.3, 0.8, 0.0])  # unknown translation we try to recover
        pos_init = jnp.zeros(3)
        params = (rf._rotation_y(76.0) @ R_true, pos_init)
        n_steps = 100
        learning_rate = 1e-2
        scale = 1.0
        # rot_init / params set below relative to R_true (180° about Y from ground truth).
    elif select_run == "sphere":
        H, W = 128, 128
        angle_true_deg = 90  # unknown rotation we try to recover
        R_true = rf._rotation_y(jnp.deg2rad(angle_true_deg)) @ rf._rotation_x(jnp.deg2rad(180))
        position_true = jnp.array([0.6, 0.6, 0.0])  # unknown translation we try to recover
        learning_rate = 1e-2
        n_steps = 160
        position_start = jnp.array([-0.6, -0.6, 0.0])
        params = (jnp.diag(jnp.array([1.3, 1.1, 0.9])), position_start)  # Init values
        learning_rate = 1e-1
        scale = 0.7
        
    sphere_v, sphere_n, sphere_t = rf._load_obj(os.path.join(here, f"{select_run}.obj"))

    camera = rf.Camera(
        fov=jnp.asarray(jnp.pi / 3),
        transform=rf.Transform(
            position=jnp.array([0.0, 0.0, -3.0]),
            rotation=jnp.eye(3),
            scale=1.0,
        ),
    )

    def empty_target():
        return rf.RenderTarget(
            color_buffer=jnp.zeros((H, W, 3)),
            depth_buffer=jnp.full((H, W), jnp.inf),
        )

    def make_scene(rotation, position):
        """Scene with the sphere at ``rotation``/``position`` over a flat white background."""
        sphere = rf.Model(
            vertices=sphere_v,
            tex_coords=sphere_t,
            normals=sphere_n,
            transform=rf.Transform(
                position=position,
                rotation=rotation,
                scale=jnp.asarray(scale, dtype=jnp.float32),
            ),
            shader=rf._normal_shader,
        )
        return rf.SceneData(camera=camera, models=[sphere])

    def render_color(rotation, position, mode):
        out = rf.render(empty_target(), make_scene(rotation, position),
                         mode=mode)
        return out.color_buffer
    
    # --- Target: sphere at an unknown pose, hard rasterization ------------
    target_image = render_color(R_true, position_true, mode="hard")

    # --- Loss: L2 between smooth render and the hard target ---------------
    # ``params`` is a (M, t) tuple: M is a free 3x3 projected to SO(3),
    # t is a free 3-vector used directly as the object translation.
    def loss_fn(params):
        M, t = params
        pred = render_color(svd(M), t, "smooth")
        return jnp.mean((pred - target_image) ** 2)

    # --- Optimizer (equinox-style jitted step) ----------------------------
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def make_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Iterations at which to snapshot the (hard) render for the progress plot.
    #snapshot_steps = [0, n_steps // 4, 2 * n_steps // 4, 3 * n_steps // 4, n_steps - 1]
    snapshot_steps = [0, 10, 20, 40, 80, 160]
    
    # --- Training loop ----------------------------------------------------
    losses, rot_errors, pos_errors = [], [], []
    snapshots = []  # (step, hard render) at each snapshot_steps entry
    print(f"Fitting pose: target rotation = {angle_true_deg:.1f} deg about Y, "
          f"target position = {tuple(float(v) for v in position_true)}, "
          f"init = (I, 0), lr = {learning_rate}, steps = {n_steps}")
    for step in range(n_steps):
        print(f"Step {step:4d}/{n_steps}...", end="")
        params, opt_state, loss = make_step(params, opt_state)
        M, t = params
        err_deg = float(jnp.rad2deg(geodesic_angle(svd(M), R_true)))
        pos_err = float(jnp.linalg.norm(t - position_true))
        losses.append(float(loss))
        rot_errors.append(err_deg)
        pos_errors.append(pos_err)
        if step in snapshot_steps:
            snapshots.append((step, render_color(svd(M), t, mode="hard")))
        if step % 20 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}   loss {float(loss):.6e}   "
                  f"rot_err {err_deg:7.3f} deg   pos_err {pos_err:.4f}")

    M_est, t_est = params
    R_est = svd(M_est)
    est_deg = float(jnp.rad2deg(y_angle(R_est)))
    print(f"Done. estimated Y-angle = {est_deg:.2f} deg "
          f"(target {angle_true_deg:.1f} deg), "
          f"estimated position = {tuple(float(v) for v in t_est)} "
          f"(target {tuple(float(v) for v in position_true)}), "
          f"final rotation error = {rot_errors[-1]:.3f} deg, "
          f"final position error = {pos_errors[-1]:.4f}")

    # --- Plots ------------------------------------------------------------
    init_image = render_color(jnp.eye(3), jnp.zeros(3), mode="hard")
    est_image = render_color(R_est, t_est, mode="hard")

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Rigid pose fitting via the differentiable rasterizer")
    gs = fig.add_gridspec(2, 3)

    panels = [
        (f"target (R={angle_true_deg:.0f} deg, t={tuple(float(v) for v in position_true)}, hard)",
         target_image),
        ("initial (R=I, t=0, hard)", init_image),
        (f"estimated (R={est_deg:.1f} deg, t={tuple(round(float(v), 2) for v in t_est)}, hard)",
         est_image),
    ]
    for col, (title, img) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(jnp.clip(img, 0.0, 1.0))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_loss = fig.add_subplot(gs[1, 0])
    ax_loss.semilogy(losses, color="C0")
    ax_loss.set_xlabel("iteration")
    #ax_loss.set_ylabel("L2 image loss")
    ax_loss.set_title("Euclidean image loss", fontsize=10)
    ax_loss.grid(True, alpha=0.3)

    ax_rot = fig.add_subplot(gs[1, 1])
    ax_rot.plot(rot_errors, color="C1")
    ax_rot.set_xlabel("iteration")
    ax_rot.set_ylabel("geodesic error (deg)")
    ax_rot.set_title("rotation error", fontsize=10)
    ax_rot.grid(True, alpha=0.3)

    ax_pos = fig.add_subplot(gs[1, 2])
    ax_pos.plot(pos_errors, color="C2")
    ax_pos.set_xlabel("iteration")
    ax_pos.set_ylabel("||t - t_true||")
    ax_pos.set_title("position error", fontsize=10)
    ax_pos.grid(True, alpha=0.3)

    out_path = os.path.join(here, "pose_fitting.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")

    # --- Progress plot: loss | target | 4 estimated renders ---------------
    fig2, axes2 = plt.subplots(1, 7, figsize=(2.4 * 6, 3.0),
                               constrained_layout=True)
    #fig2.suptitle("Pose fitting progress")

    axes2[0].semilogy(losses, color="C0")
    axes2[0].set_xlabel("iteration")
    axes2[0].set_ylabel("L2 image loss")
    #axes2[0].set_title("Euclidean image loss", fontsize=9)
    axes2[0].grid(True, alpha=0.3)

    axes2[1].imshow(jnp.clip(target_image, 0.0, 1.0))
    axes2[1].set_title("target image", fontsize=9)
    axes2[1].set_xticks([])
    axes2[1].set_yticks([])

    for ax, (snap_step, snap_img) in zip(axes2[2:], snapshots):
        ax.imshow(jnp.clip(snap_img, 0.0, 1.0))
        ax.set_title(f"estimate - iteration {snap_step}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    out_path2 = os.path.join(here, "pose_fitting_progress.png")
    fig2.savefig(out_path2, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_path2}")


if __name__ == "__main__":
    main()
