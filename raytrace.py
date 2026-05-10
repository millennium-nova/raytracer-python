import math
import os
import sys
from enum import Enum

import numpy as np


class Material(Enum):
    DIFFUSE = 1
    METAL = 2
    DIELECTRIC = 3


np.random.seed(0)

preview_mode = ("--preview" in sys.argv) or os.getenv("RAYTRACE_PREVIEW", "0") == "1"


# ============================================================
# Scene
# ============================================================

spheres = [
    # ground
    (np.array([0.0, -1000.0, 0.0]), 1000.0,
     np.array([0.48, 0.50, 0.52]), Material.DIFFUSE, 0.0),

    # red diffuse sphere
    (np.array([-1.15, 0.95, -0.35]), 0.95,
     np.array([0.82, 0.28, 0.20]), Material.DIFFUSE, 0.0),

    # center perfect mirror sphere
    # 完全鏡面に近い中央球
    (np.array([0.35, 1.05, 0.05]), 1.05,
     np.array([1.00, 1.00, 1.00]), Material.METAL, 0.0),

    # right greenish glossy metal sphere
    # 緑色っぽい金属光沢の右球
    (np.array([2.05, 0.78, 0.25]), 0.78,
     np.array([0.34, 0.68, 0.52]), Material.METAL, 0.03),

    # front green accent
    (np.array([1.05, 0.38, -1.15]), 0.38,
     np.array([0.22, 0.68, 0.30]), Material.DIFFUSE, 0.0),

    # small left gold metal accent
    (np.array([-2.65, 0.42, 0.30]), 0.42,
     np.array([0.90, 0.76, 0.46]), Material.METAL, 0.08),
]


light_sources = [
    # warm key light
    (np.array([-4.0, 7.0, 5.0]), np.array([1.0, 0.86, 0.68]), 1.8),

    # cool fill light
    (np.array([5.0, 5.5, 4.0]), np.array([0.55, 0.68, 1.0]), 0.55),
]

ambient_light = np.array([0.08, 0.09, 0.11])


# ============================================================
# Utilities
# ============================================================

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def reflect(v, n):
    return v - 2.0 * np.dot(v, n) * n


def refract(uv, n, eta_ratio):
    cos_theta = min(np.dot(-uv, n), 1.0)
    r_out_perp = eta_ratio * (uv + cos_theta * n)
    k = 1.0 - np.dot(r_out_perp, r_out_perp)
    if k < 0.0:
        return None
    r_out_parallel = -math.sqrt(k) * n
    return r_out_perp + r_out_parallel


def schlick(cosine, ref_idx):
    r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)) ** 2
    return r0 + (1.0 - r0) * ((1.0 - cosine) ** 5)


def random_in_unit_sphere():
    while True:
        p = np.random.uniform(-1.0, 1.0, 3)
        if np.dot(p, p) < 1.0:
            return p


def sky_color(direction):
    d = normalize(direction)
    t = 0.5 * (d[1] + 1.0)

    bottom = np.array([0.86, 0.89, 0.94])
    top = np.array([0.48, 0.66, 0.88])

    return (1.0 - t) * bottom + t * top


# ============================================================
# Intersection
# ============================================================

def hit_sphere(origin, direction, sphere, t_min, t_max):
    center, radius, color, material, param = sphere

    oc = origin - center
    a = np.dot(direction, direction)
    half_b = np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius

    disc = half_b * half_b - a * c
    if disc < 0.0:
        return None

    sqrtd = math.sqrt(disc)

    root = (-half_b - sqrtd) / a
    if root < t_min or root > t_max:
        root = (-half_b + sqrtd) / a
        if root < t_min or root > t_max:
            return None

    pos = origin + root * direction
    outward_normal = normalize(pos - center)

    front_face = np.dot(direction, outward_normal) < 0.0
    normal = outward_normal if front_face else -outward_normal

    return {
        "t": root,
        "pos": pos,
        "normal": normal,
        "front_face": front_face,
        "color": color,
        "material": material,
        "param": param,
    }


def intersect(origin, direction, t_min=1e-4, t_max=float("inf")):
    closest = t_max
    hit = None

    for sphere in spheres:
        h = hit_sphere(origin, direction, sphere, t_min, closest)
        if h is not None:
            closest = h["t"]
            hit = h

    return hit


# ============================================================
# Shading
# ============================================================

def direct_lighting(hit, view_dir):
    pos = hit["pos"]
    normal = hit["normal"]
    color = hit["color"]
    material = hit["material"]
    param = hit["param"]

    result = ambient_light * color

    for light_pos, light_color, intensity in light_sources:
        to_light = light_pos - pos
        dist = np.linalg.norm(to_light)
        light_dir = to_light / dist

        shadow = intersect(pos + normal * 1e-4, light_dir, 1e-4, dist - 1e-4)
        if shadow is not None:
            continue

        ndotl = max(0.0, np.dot(normal, light_dir))
        attenuation = intensity / (0.18 * dist * dist)

        if material == Material.DIFFUSE:
            result += color * light_color * ndotl * attenuation

        elif material == Material.METAL:
            half_dir = normalize(light_dir - view_dir)
            spec = max(0.0, np.dot(normal, half_dir))

            # param は粗さ。0 に近いほど鋭いハイライト
            roughness = max(0.02, param)
            shininess = 120.0 / roughness

            result += color * light_color * (
                0.14 * ndotl + 0.58 * (spec ** shininess)
            ) * attenuation

        elif material == Material.DIELECTRIC:
            half_dir = normalize(light_dir - view_dir)
            spec = max(0.0, np.dot(normal, half_dir)) ** 96
            result += light_color * (0.04 * ndotl + 1.35 * spec) * attenuation

    return result


def trace(origin, direction, depth):
    if depth <= 0:
        return np.array([0.0, 0.0, 0.0])

    hit = intersect(origin, direction)
    if hit is None:
        return sky_color(direction)

    pos = hit["pos"]
    normal = hit["normal"]
    color = hit["color"]
    material = hit["material"]
    param = hit["param"]

    result = direct_lighting(hit, direction)

    if material == Material.DIFFUSE:
        # preview では間接光を切って軽量・低ノイズにする
        if not preview_mode:
            scatter = normalize(normal + normalize(random_in_unit_sphere()))
            indirect = trace(pos + normal * 1e-4, scatter, depth - 1)
            result += 0.16 * color * indirect

    elif material == Material.METAL:
        # 安定重視: 金属反射ではランダム散乱しない
        reflected = normalize(reflect(normalize(direction), normal))

        if np.dot(reflected, normal) > 0.0:
            # roughness に応じて反射寄与を少し下げる
            reflection_strength = 0.78 * (1.0 - min(param, 0.8) * 0.45)
            result += reflection_strength * color * trace(
                pos + normal * 1e-4,
                reflected,
                depth - 1
            )

    elif material == Material.DIELECTRIC:
        # 今回のシーンでは未使用。残しておくが、中央球には使わない。
        ref_idx = param
        eta_ratio = 1.0 / ref_idx if hit["front_face"] else ref_idx

        unit_dir = normalize(direction)
        cos_theta = min(np.dot(-unit_dir, normal), 1.0)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

        reflected = normalize(reflect(unit_dir, normal))
        reflected_col = trace(pos + normal * 1e-4, reflected, depth - 1)

        cannot_refract = eta_ratio * sin_theta > 1.0
        fresnel = schlick(cos_theta, ref_idx)

        if cannot_refract:
            result += 0.90 * reflected_col
        else:
            refracted = refract(unit_dir, normal, eta_ratio)
            if refracted is None:
                result += 0.90 * reflected_col
            else:
                refracted = normalize(refracted)

                if hit["front_face"]:
                    refract_origin = pos - normal * 1e-4
                else:
                    refract_origin = pos + normal * 1e-4

                refracted_col = trace(refract_origin, refracted, depth - 1)
                attenuation = np.array([0.96, 0.98, 1.0])

                result += 0.88 * attenuation * (
                    fresnel * reflected_col + (1.0 - fresnel) * refracted_col
                )

    return np.clip(result, 0.0, 1.0)


# ============================================================
# Camera
# ============================================================

if preview_mode:
    W, H = 320, 180
    samples = 2
    max_depth = 3
else:
    W, H = 800, 450
    samples = 24
    max_depth = 8

fov = 32.0

camera_origin = np.array([4.2, 1.75, 5.2])
camera_target = np.array([0.15, 0.75, -0.10])

forward = normalize(camera_target - camera_origin)
right = normalize(np.cross(forward, np.array([0.0, 1.0, 0.0])))
up = normalize(np.cross(right, forward))

aspect = W / H
viewport_h = 2.0 * math.tan(math.radians(fov) * 0.5)
viewport_w = aspect * viewport_h


# ============================================================
# Render
# ============================================================

print(f"rendering: {W}x{H}, samples={samples}, depth={max_depth}, preview={preview_mode}")

with open("render.ppm", "w") as f:
    f.write(f"P3\n{W} {H}\n255\n")

    for y in range(H):
        if y % 40 == 0:
            print(f"{y}/{H}")

        for x in range(W):
            pixel = np.array([0.0, 0.0, 0.0])

            for _ in range(samples):
                u = (x + np.random.random()) / (W - 1)
                v = (y + np.random.random()) / (H - 1)

                px = (u - 0.5) * viewport_w
                py = -(v - 0.5) * viewport_h

                ray_dir = normalize(forward + px * right + py * up)
                pixel += trace(camera_origin, ray_dir, max_depth)

            pixel /= samples

            # gamma correction
            pixel = np.power(np.clip(pixel, 0.0, 1.0), 1.0 / 2.2)

            r, g, b = (255.999 * np.clip(pixel, 0.0, 0.999)).astype(int)
            f.write(f"{r} {g} {b} ")

        f.write("\n")

print("done: render.ppm")