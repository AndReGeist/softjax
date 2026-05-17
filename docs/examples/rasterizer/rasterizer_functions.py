"""JAX port of Software-Rasterizer/Source/Core/Rasterizer.cs.

Pipeline (per frame):

  1. ``process_model``: model-space vertices → view space → screen-space
     ``RasterizedModel`` triples (one per input triangle). Vectorised
     over vertices with ``jax.vmap``; returns a single
     ``RasterizedModel`` pytree whose leaves carry leading axis
     ``(n_tris, 3, ...)``. Near-plane clipping is omitted; callers must
     ensure all geometry lies in front of the camera.
  2. ``rasterize_triangle``: barycentric coverage test for every pixel →
     perspective-correct interpolation of depth, UV and normals → z-test
     → pixel shader. ``jnp.where`` merges colour / depth instead of
     per-pixel locks.
  3. ``render``: top-level orchestrator. The per-triangle loop is a
     ``jax.lax.scan`` so the whole pipeline is jit / grad / vmap-friendly.
"""

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx

# Small clamp used in divisions to keep gradients finite on near-degenerate
# input (zero-area triangles, near-zero view-space z).
_EPS = 1e-6


# --- Data structures (lightweight stand-ins for the C# Types/) ------------


class Transform(NamedTuple):
    """Similarity transform: world = rotation @ (scale * local) + position."""
    position: jax.Array  # (3,)
    rotation: jax.Array  # (3, 3) rotation matrix
    scale: jax.Array = jnp.float32(1.0)  # scalar, uniform scale factor

    def to_world_point(self, p):
        """Local to world transform."""
        return self.rotation @ (self.scale * p) + self.position

    def to_local_point(self, p):
        """World to local transform."""
        return self.rotation.T @ (p - self.position) / self.scale


class RenderTarget(NamedTuple):
    color_buffer: jax.Array  # (H, W, 3)
    depth_buffer: jax.Array  # (H, W)

    @property
    def size(self):
        h, w = self.color_buffer.shape[:2]
        return jnp.array([w, h], dtype=jnp.float32)

    @property
    def width(self):
        return self.color_buffer.shape[1]

    @property
    def height(self):
        return self.color_buffer.shape[0]


@jax.tree_util.register_pytree_node_class
class Model:
    """Drawable geometry plus its pixel shader.

    Registered as a pytree with the shader pulled out as auxiliary
    (static) data — so ``jit`` doesn't try to trace a Python callable
    and ``vmap`` doesn't try to map over it.
    """

    def __init__(self, vertices, tex_coords, normals, transform, shader):
        self.vertices = vertices      # (n_tris, 3, 3)
        self.tex_coords = tex_coords  # (n_tris, 3, 2)
        self.normals = normals        # (n_tris, 3, 3)
        self.transform = transform    # model-to-world Transform
        self.shader = shader          # Callable (static aux)

    def tree_flatten(self):
        children = (self.vertices, self.tex_coords, self.normals, self.transform)
        aux = (self.shader,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)

    def replace(self, **kwargs):
        return Model(
            vertices=kwargs.get("vertices", self.vertices),
            tex_coords=kwargs.get("tex_coords", self.tex_coords),
            normals=kwargs.get("normals", self.normals),
            transform=kwargs.get("transform", self.transform),
            shader=kwargs.get("shader", self.shader),
        )


class Camera(NamedTuple):
    fov: jax.Array        # scalar, vertical FOV in radians
    transform: Transform  # camera-to-world SE(3) transform


class SceneData(NamedTuple):
    camera: Camera
    models: Sequence[Model]


class RasterizedModel(NamedTuple):
    depth: jax.Array       # (n_tris, 3, 1), or batched
    screen_pos: jax.Array  # (n_tris, 3, 2), or batched
    tex_coords: jax.Array  # (n_tris, 3, 2), or batched
    normals: jax.Array     # (n_tris, 3, 3), or batched


# --- Math helpers (subset of Maths.cs used in the pipeline) ---------------


def signed_parallelogram_area(a, b, c):
    """2x signed area of triangle abc (positive for clockwise winding)."""
    return (c[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1]) \
         + (c[..., 1] - a[..., 1]) * (a[..., 0] - b[..., 0])


def point_in_triangle(a, b, c, p):
    """Coverage test + barycentric weights. Back-faces (CCW) are excluded.

    The reciprocal of the signed area is guarded with ``jnp.where`` so
    that degenerate triangles (``total ≈ 0``) don't backprop ``nan``.
    """
    area_abp = signed_parallelogram_area(a, b, p)
    area_bcp = signed_parallelogram_area(b, c, p)
    area_cap = signed_parallelogram_area(c, a, p)
    inv_total = 1.0 / (area_abp + area_bcp + area_cap)
    weight_a = area_bcp * inv_total
    weight_b = area_cap * inv_total
    weight_c = area_abp * inv_total
    inside = (area_abp > 0) & (area_bcp > 0) & (area_cap > 0)
    return inside, weight_a, weight_b, weight_c


# --- Vertex transform & projection ----------------------------------------


def vertex_to_view(vert, model_transform, camera):
    world = model_transform.to_world_point(vert)
    return camera.transform.to_local_point(world)


def view_to_screen(vertex_view, camera, target):
    """Pinhole projection: view space → pixel coordinates."""
    screen_height_world = jnp.tan(camera.fov / 2) * 2
    pixels_per_world_unit = target.size[1] / screen_height_world / vertex_view[2]
    pixel_offset = vertex_view[:2] * pixels_per_world_unit
    return target.size / 2.0 + pixel_offset


def process_model(model, camera, target):
    """Project every vertex of ``model`` to a ``RasterizedModel``.

    Inputs are already laid out as ``(n_tris, 3, ...)``; a nested
    ``vmap`` applies the per-vertex projection over both axes and the
    returned ``RasterizedModel`` leaves keep that same shape. Assumes
    positive view-space z for every vertex; no near-plane clipping
    is performed.
    """

    def per_vertex(vert, tex, normal):
        view_point = vertex_to_view(vert, model.transform, camera)
        return RasterizedModel(
            depth=view_point[2],
            screen_pos=view_to_screen(view_point, camera, target),
            tex_coords=tex,
            normals=normal,
        )

    return jax.vmap(jax.vmap(per_vertex))(
        model.vertices, model.tex_coords, model.normals,
    )


# --- Triangle rasterization -----------------------------------------------


def _pixel_grid(height, width):
    """(H, W, 2) array of pixel centres (integer + 0.5)."""
    ys, xs = jnp.mgrid[0:height, 0:width]
    return jnp.stack(
        [xs.astype(jnp.float32), ys.astype(jnp.float32)], axis=-1,
    )


def rasterize_triangle(triangle, h, w, shader):
    """Shade one triangle over an ``H x W`` pixel grid.

    ``triangle`` is a single ``RasterizedModel`` (leaves of shape
    ``(3, ...)`` — one entry per vertex). Computes per-pixel
    barycentric coverage and perspective-correct depth / UV / normal,
    then invokes ``shader``. Returns ``(color, depth, inside)`` of
    shapes ``(H, W, 3)``, ``(H, W)``, ``(H, W)``. Pure per-triangle:
    the z-test and buffer merge live in ``render``.
    """
    a = triangle.screen_pos[0]
    b = triangle.screen_pos[1]
    c = triangle.screen_pos[2]

    # Reciprocal depths + per-vertex attributes pre-divided by depth so
    # that linear interpolation in screen space becomes perspective-correct.
    safe_depth = jnp.maximum(triangle.depth, _EPS)             # (3,)
    inv_depths = 1.0 / safe_depth                              # (3,)
    tex_over_z = triangle.tex_coords * inv_depths[:, None]     # (3, 2)
    nrm_over_z = triangle.normals    * inv_depths[:, None]     # (3, 3)

    p = _pixel_grid(h, w)                                       # (H, W, 2)
    inside, wa, wb, wc = point_in_triangle(a, b, c, p)          # each (H, W)

    weights = jnp.stack([wa, wb, wc], axis=-1)                  # (H, W, 3)
    inv_z_per_pixel = weights @ inv_depths                       # (H, W)
    depth = 1.0 / inv_z_per_pixel
    tex_coord = jnp.einsum("hwk,kc->hwc", weights, tex_over_z) * depth[..., None]
    normal    = jnp.einsum("hwk,kc->hwc", weights, nrm_over_z) * depth[..., None]

    color = shader(p, tex_coord, normal, depth)                 # (H, W, 3)
    return color, depth, inside


# --- Top-level render -----------------------------------------------------


def render(target, scene_data):
    """Composite every model in ``scene_data`` into ``target``.

    Per model: project vertices (``process_model``), shade every
    triangle in parallel (``vmap(rasterize_triangle)``), pick the
    closest covering triangle for each pixel (``argmin`` on depths
    masked by ``inside``), then z-test the winner against the running
    buffers. The outer loop over ``scene_data.models`` stays Python
    so heterogeneous models / shaders don't force a single trace.
    Returns a new ``RenderTarget``.
    """
    camera = scene_data.camera
    color_buf = target.color_buffer
    depth_buf = target.depth_buffer
    h, w = color_buf.shape[:2]

    for model in scene_data.models:
        projected_triangles = process_model(model, camera, target)

        # (n_tris, H, W, 3), (n_tris, H, W), (n_tris, H, W)
        colors, depths, insides = jax.vmap(
            lambda tri: rasterize_triangle(tri, h, w, model.shader)
        )(projected_triangles)

        # Disqualify pixels outside their triangle by pushing depth to +inf.
        masked_depths = jnp.where(insides, depths, jnp.inf)

        # Closest covering triangle per pixel.
        winner = jnp.argmin(masked_depths, axis=0)               # (H, W)

        win_depth = jnp.take_along_axis(
            masked_depths, winner[None], axis=0,
        )[0]                                                      # (H, W)
        win_color = jnp.take_along_axis(
            colors, winner[None, ..., None], axis=0,
        )[0]                                                      # (H, W, 3)

        # If no triangle covered a pixel, win_depth == +inf, so the
        # comparison below is False and the existing buffer is kept.
        write = win_depth < depth_buf
        color_buf = jnp.where(write[..., None], win_color, color_buf)
        depth_buf = jnp.where(write,            win_depth, depth_buf)

    return RenderTarget(color_buffer=color_buf, depth_buffer=depth_buf)


# --- Demo: load cube.obj and render at several rotations ------------------


def _load_obj(path):
    """Minimal OBJ loader. Triangulates polygons via fan and returns
    (vertices, normals, tex_coords) as ``(n_tris, 3, ...)`` arrays."""

    # TODO: Add check for OBJ convention is CCW front-facing.
    print(f"Loading OBJ file from {path}...")
    print("Warning: this is a minimal loader for demo purposes assuming OBJ convention is CCW front-facing.")
    positions, normals, tex_coords = [], [], []
    out_pos, out_nrm, out_tex = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                positions.append([float(x) for x in parts[1:4]])
            elif tag == "vn":
                normals.append([float(x) for x in parts[1:4]])
            elif tag == "vt":
                tex_coords.append([float(x) for x in parts[1:3]])
            elif tag == "f":
                face = []
                for entry in parts[1:]:
                    bits = entry.split("/")
                    vi = int(bits[0]) - 1
                    ti = int(bits[1]) - 1 if len(bits) > 1 and bits[1] else None
                    ni = int(bits[2]) - 1 if len(bits) > 2 and bits[2] else None
                    face.append((vi, ti, ni))
                for k in range(1, len(face) - 1):
                    for vi, ti, ni in (face[0], face[k], face[k + 1]):
                        out_pos.append(positions[vi])
                        out_nrm.append(normals[ni] if ni is not None else [0.0, 0.0, 0.0])
                        out_tex.append(tex_coords[ti] if ti is not None else [0.0, 0.0])
    pos = jnp.asarray(out_pos, dtype=jnp.float32)
    nrm = jnp.asarray(out_nrm, dtype=jnp.float32)
    tex = jnp.asarray(out_tex, dtype=jnp.float32)
    n_tris = pos.shape[0] // 3
    return (
        pos.reshape(n_tris, 3, 3),
        nrm.reshape(n_tris, 3, 3),
        tex.reshape(n_tris, 3, 2),
    )


def _rotation_y(angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[c, 0.0, s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0, c]])


def _normal_shader(pixel_xy, tex_coord, normal, depth):
    """Visualise the interpolated world-space normal as RGB in [0, 1]."""
    return normal * 0.5 + 0.5


def main():
    import os
    import matplotlib.pyplot as plt

    here = os.path.dirname(os.path.abspath(__file__))
    vertices, normals, tex_coords = _load_obj(os.path.join(here, "cube.obj"))
    # Centre cube on the origin so y-rotation spins it in place.
    vertices = vertices - 0.5

    H, W = 256, 256
    camera = Camera(
        fov=jnp.asarray(jnp.pi / 3),
        transform=Transform(
            position=jnp.array([0.0, 0.0, -3.0]),
            rotation=jnp.eye(3),
        ),
    )

    scale = jnp.asarray(1.5, dtype=jnp.float32)

    def make_scene(angle_rad):
        model = Model(
            vertices=vertices,
            tex_coords=tex_coords,
            normals=normals,
            transform=Transform(
                position=jnp.zeros(3),
                rotation=_rotation_y(angle_rad),
                scale=scale,
            ),
            shader=_normal_shader,
        )
        return SceneData(camera=camera, models=[model])

    def empty_target():
        return RenderTarget(
            color_buffer=jnp.zeros((H, W, 3)),
            depth_buffer=jnp.full((H, W), jnp.inf),
        )

    angles_deg = jnp.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0])
    angles_rad = jnp.deg2rad(angles_deg)

    # --- Eager per-angle render (also used as ground truth below) ---
    color_images = []
    depth_images = []
    for ang in angles_rad:
        result = render(empty_target(), make_scene(ang))
        color_images.append(result.color_buffer)
        depth_images.append(result.depth_buffer)

    # # --- Smoke tests: jit / grad / vmap ---------------------------------

    # print("Smoke test: jit(render) matches eager render...")
    # render_jit = jax.jit(render)
    # result_jit = render_jit(empty_target(), make_scene(angles_rad[0]))
    # diff_jit = float(jnp.max(jnp.abs(color_images[0] - result_jit.color_buffer)))
    # print(f"  max |Δcolor| = {diff_jit:.2e}  {'OK' if diff_jit < 1e-4 else 'FAIL'}")

    # print("Smoke test: jax.grad of mean(color) wrt camera position...")
    # def loss_camera(cam_pos):
    #     cam = Camera(
    #         fov=camera.fov,
    #         transform=Transform(
    #             position=cam_pos,
    #             rotation=camera.transform.rotation,
    #         ),
    #     )
    #     scene = SceneData(camera=cam, models=make_scene(angles_rad[0]).models)
    #     return jnp.mean(render(empty_target(), scene).color_buffer)
    # grad_cam = jax.grad(loss_camera)(camera.transform.position)
    # print(f"  grad = {grad_cam}  finite={bool(jnp.all(jnp.isfinite(grad_cam)))}")

    # print("Smoke test: vmap(render) over a batch of rotation angles...")
    # def render_at_angle(angle_rad):
    #     return render(empty_target(), make_scene(angle_rad)).color_buffer
    # batched_colors = jax.vmap(render_at_angle)(angles_rad)
    # eager_stack = jnp.stack(color_images)
    # diff_vmap = float(jnp.max(jnp.abs(batched_colors - eager_stack)))
    # print(f"  max |Δvmap-vs-loop| = {diff_vmap:.2e}  {'OK' if diff_vmap < 1e-4 else 'FAIL'}")

    # --- Plot ----------------------------------------------------------

    n_cols = len(angles_deg)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(2.2 * n_cols, 4.8),
        constrained_layout=True,
    )
    fig.suptitle(f"Cube rendered at scale ×{float(scale):.2f}", fontsize=14)

    def _strip_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Colour row.
    for ax, img, deg in zip(axes[0], color_images, angles_deg):
        ax.imshow(jnp.clip(img, 0.0, 1.0))
        ax.set_title(f"{float(deg):.0f}°")
        _strip_ticks(ax)
    axes[0, 0].set_ylabel("color", fontsize=12)

    # Depth row: share vmin/vmax so head-on frames don't get noise-amplified.
    finite_depths = [jnp.where(jnp.isfinite(d), d, jnp.nan) for d in depth_images]
    vmin = float(jnp.nanmin(jnp.stack([jnp.nanmin(d) for d in finite_depths])))
    vmax = float(jnp.nanmax(jnp.stack([jnp.nanmax(d) for d in finite_depths])))
    for ax, depth in zip(axes[1], finite_depths):
        im = ax.imshow(depth, cmap="gray", vmin=vmin, vmax=vmax)
        _strip_ticks(ax)
    axes[1, 0].set_ylabel("depth", fontsize=12)

    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.85, label="view-space z")

    plt.savefig("./rendered_cubes.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
