"""JAX port of Software-Rasterizer/Source/Core/Rasterizer.cs.

Pipeline (per frame):

  1. ``process_model``: model-space vertices → view space, then per-triangle
     near-plane clipping that can split one triangle into 0, 1, or 2 output
     triangles. Output is a flat list of ``RasterizerPoint`` triples.
  2. ``rasterize_triangle``: screen-space AABB → barycentric coverage test
     for every pixel → perspective-correct interpolation of depth, UV and
     normals → z-test → pixel shader. The C# version uses bbox loops with
     per-pixel locks; here we compute coverage for the whole framebuffer
     and merge with ``jnp.where``.
  3. ``render``: top-level orchestrator.
"""

from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp

# Pixel shader signature: (pixel_xy, tex_coord, normal, depth) -> rgb.
# All inputs are broadcast over the framebuffer (shape (H, W, ...)).
Shader = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]


# --- Data structures (lightweight stand-ins for the C# Types/) ------------


class Transform(NamedTuple):
    """Affine transform: world = rotation @ local + position."""
    position: jax.Array  # (3,)
    rotation: jax.Array  # (3, 3) rotation matrix

    def to_world_point(self, p):
        return self.rotation @ p + self.position

    def to_local_point(self, p):
        return self.rotation.T @ (p - self.position)


class Camera(NamedTuple):
    fov: jax.Array       # scalar, vertical FOV in radians
    transform: Transform


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


class Model(NamedTuple):
    vertices: jax.Array    # (3 * n_tris, 3)
    tex_coords: jax.Array  # (3 * n_tris, 2)
    normals: jax.Array     # (3 * n_tris, 3)
    transform: Transform
    shader: Shader


class SceneData(NamedTuple):
    camera: Camera
    models: Sequence[Model]


class RasterizerPoint(NamedTuple):
    depth: jax.Array       # scalar
    screen_pos: jax.Array  # (2,)
    tex_coords: jax.Array  # (2,)
    normals: jax.Array     # (3,)


# --- Math helpers (subset of Maths.cs used in the pipeline) ---------------


def signed_parallelogram_area(a, b, c):
    """2x signed area of triangle abc (positive for clockwise winding)."""
    return (c[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1]) \
         + (c[..., 1] - a[..., 1]) * (a[..., 0] - b[..., 0])


def point_in_triangle(a, b, c, p):
    """Coverage test + barycentric weights. Back-faces (CCW) are excluded."""
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


def lerp(a, b, t):
    return a + (b - a) * t


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


# --- Near-plane clipping --------------------------------------------------

NEAR_CLIP_DST = 0.01


def _emit_point(view_point, tex, normal, camera, target):
    return RasterizerPoint(
        depth=view_point[2],
        screen_pos=view_to_screen(view_point, camera, target),
        tex_coords=tex,
        normals=normal,
    )


def _edge_crossing(view_pts, texs, norms, i_anchor, i_other, near):
    """Interpolate (view, tex, normal) along an edge to the near plane."""
    z_anchor = view_pts[i_anchor][2]
    z_other = view_pts[i_other][2]
    t = (near - z_anchor) / (z_other - z_anchor)
    return (
        lerp(view_pts[i_anchor], view_pts[i_other], t),
        lerp(texs[i_anchor], texs[i_other], t),
        lerp(norms[i_anchor], norms[i_other], t),
    )


def _clip_triangle(view_pts, texs, norms, camera, target):
    """Near-plane clip one triangle; returns 0, 1, or 2 output triangles.

    Mirrors the four-case switch in Rasterizer.cs::ProcessModel.
    """
    clipped = [bool(v[2] <= NEAR_CLIP_DST) for v in view_pts]
    n_clipped = sum(clipped)

    if n_clipped == 3:
        return []

    if n_clipped == 0:
        return [tuple(
            _emit_point(view_pts[i], texs[i], norms[i], camera, target)
            for i in range(3)
        )]

    if n_clipped == 1:
        i_clip = clipped.index(True)
        i_next = (i_clip + 1) % 3
        i_prev = (i_clip - 1) % 3
        edge_a_v, edge_a_t, edge_a_n = _edge_crossing(
            view_pts, texs, norms, i_clip, i_next, NEAR_CLIP_DST)
        edge_b_v, edge_b_t, edge_b_n = _edge_crossing(
            view_pts, texs, norms, i_clip, i_prev, NEAR_CLIP_DST)
        edge_a = _emit_point(edge_a_v, edge_a_t, edge_a_n, camera, target)
        edge_b = _emit_point(edge_b_v, edge_b_t, edge_b_n, camera, target)
        v_next = _emit_point(view_pts[i_next], texs[i_next], norms[i_next], camera, target)
        v_prev = _emit_point(view_pts[i_prev], texs[i_prev], norms[i_prev], camera, target)
        return [(edge_b, edge_a, v_prev),
                (edge_a, v_next, v_prev)]

    # n_clipped == 2 — only the lone surviving vertex remains.
    i_keep = clipped.index(False)
    i_next = (i_keep + 1) % 3
    i_prev = (i_keep - 1) % 3
    edge_a_v, edge_a_t, edge_a_n = _edge_crossing(
        view_pts, texs, norms, i_keep, i_next, NEAR_CLIP_DST)
    edge_b_v, edge_b_t, edge_b_n = _edge_crossing(
        view_pts, texs, norms, i_keep, i_prev, NEAR_CLIP_DST)
    edge_a = _emit_point(edge_a_v, edge_a_t, edge_a_n, camera, target)
    edge_b = _emit_point(edge_b_v, edge_b_t, edge_b_n, camera, target)
    v_keep = _emit_point(view_pts[i_keep], texs[i_keep], norms[i_keep], camera, target)
    return [(edge_b, v_keep, edge_a)]


def process_model(model, camera, target):
    """Transform vertices and produce a flat list of clipped rasterizer triangles."""
    n_verts = model.vertices.shape[0]
    out = []
    for i in range(0, n_verts, 3):
        view_pts = [vertex_to_view(model.vertices[i + k], model.transform, camera)
                    for k in range(3)]
        texs = [model.tex_coords[i + k] for k in range(3)]
        norms = [model.normals[i + k] for k in range(3)]
        out.extend(_clip_triangle(view_pts, texs, norms, camera, target))
    return out


# --- Triangle rasterization -----------------------------------------------


def _pixel_grid(height, width):
    ys, xs = jnp.mgrid[0:height, 0:width]
    return jnp.stack(
        [xs.astype(jnp.float32), ys.astype(jnp.float32)], axis=-1,
    )  # (H, W, 2)


def rasterize_triangle(rp0, rp1, rp2, color_buf, depth_buf, shader):
    """Rasterize one triangle and return updated (color_buf, depth_buf).

    Equivalent to the inner Parallel. For body in Rasterizer.cs::Render, but
    without per-pixel locks: depth + color merge is done with ``jnp.where``.
    """
    h, w = color_buf.shape[:2]
    a, b, c = rp0.screen_pos, rp1.screen_pos, rp2.screen_pos

    # Reciprocal depths and per-vertex attributes pre-divided by depth, so
    # that linear interpolation in screen space becomes perspective-correct.
    inv_depths = jnp.stack([1.0 / rp0.depth, 1.0 / rp1.depth, 1.0 / rp2.depth])
    tex_over_z = jnp.stack([rp0.tex_coords * inv_depths[0],
                            rp1.tex_coords * inv_depths[1],
                            rp2.tex_coords * inv_depths[2]])  # (3, 2)
    nrm_over_z = jnp.stack([rp0.normals * inv_depths[0],
                            rp1.normals * inv_depths[1],
                            rp2.normals * inv_depths[2]])     # (3, 3)

    p = _pixel_grid(h, w)  # (H, W, 2)
    inside, wa, wb, wc = point_in_triangle(a, b, c, p)         # each (H, W)

    # depth = 1 / interp(1/z)
    depth = 1.0 / (inv_depths[0] * wa + inv_depths[1] * wb + inv_depths[2] * wc)

    # attr = interp(attr / z) * depth
    weights = jnp.stack([wa, wb, wc], axis=-1)                 # (H, W, 3)
    tex_coord = jnp.einsum("hwk,kc->hwc", weights, tex_over_z) * depth[..., None]
    normal    = jnp.einsum("hwk,kc->hwc", weights, nrm_over_z) * depth[..., None]

    new_color = shader(p, tex_coord, normal, depth)            # (H, W, 3)

    write = inside & (depth < depth_buf)
    color_out = jnp.where(write[..., None], new_color, color_buf)
    depth_out = jnp.where(write, depth, depth_buf)
    return color_out, depth_out


# --- Top-level render -----------------------------------------------------


def render(target, scene_data):
    """Render ``scene_data`` into ``target``; returns a new RenderTarget."""
    camera = scene_data.camera
    color_buf = target.color_buffer
    depth_buf = target.depth_buffer

    for model in scene_data.models:
        for rp0, rp1, rp2 in process_model(model, camera, target):
            color_buf, depth_buf = rasterize_triangle(
                rp0, rp1, rp2, color_buf, depth_buf, model.shader,
            )

    return RenderTarget(color_buffer=color_buf, depth_buffer=depth_buf)


# --- Demo: load cube.obj and render at several rotations ------------------


def _load_obj(path):
    """Minimal OBJ loader. Triangulates polygons via fan and returns
    (vertices, normals, tex_coords) as ``(3 * n_tris, ...)`` arrays."""
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
    return (
        jnp.asarray(out_pos, dtype=jnp.float32),
        jnp.asarray(out_nrm, dtype=jnp.float32),
        jnp.asarray(out_tex, dtype=jnp.float32),
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

    angles_deg = [0, 60, 120, 180, 240, 300]
    images = []
    for deg in angles_deg:
        model = Model(
            vertices=vertices,
            tex_coords=tex_coords,
            normals=normals,
            transform=Transform(
                position=jnp.zeros(3),
                rotation=_rotation_y(jnp.deg2rad(deg)),
            ),
            shader=_normal_shader,
        )
        target = RenderTarget(
            color_buffer=jnp.zeros((H, W, 3)),
            depth_buffer=jnp.full((H, W), jnp.inf),
        )
        result = render(target, SceneData(camera=camera, models=[model]))
        images.append(result.color_buffer)

    fig, axes = plt.subplots(1, len(angles_deg), figsize=(3 * len(angles_deg), 3))
    for ax, img, deg in zip(axes, images, angles_deg):
        ax.imshow(jnp.asarray(img))
        ax.set_title(f"{deg}°")
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("./rendered_cubes.png")


if __name__ == "__main__":
    main()

