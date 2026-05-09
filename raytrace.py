import math
import numpy as np
from enum import Enum

# マテリアルタイプ
class Material(Enum):
    DIFFUSE = 1      # 拡散反射（マット）
    METAL = 2        # 金属
    DIELECTRIC = 3   # 誘電体（ガラス）

# シーン定義
# 球体: (中心座標, 半径, 色, マテリアル, 反射率/屈折率/粗さ)
# Material.DIFFUSE: 基本の色で拡散反射
# Material.METAL: 金属色、粗さパラメータで反射の鋭さを制御
# Material.DIELECTRIC: 屈折率、IOR（通常1.5ガラス）
spheres = [
    # 地面 - 暗い拡散反射
    (np.array([0, -10004, -20]), 10000, np.array([0.35, 0.35, 0.35]), Material.DIFFUSE, 0.0),
    
    # 赤い拡散反射球（左奥）
    (np.array([-8, 1.5, -22]), 2.0, np.array([0.95, 0.2, 0.15]), Material.DIFFUSE, 0.0),
    
    # 磨かれた銀球（中央奥）
    (np.array([0, 2.5, -24]), 2.2, np.array([0.95, 0.96, 1.0]), Material.METAL, 0.05),
    
    # ガラス球（中央前、見どころ）
    (np.array([1, 2.8, -18]), 2.5, np.array([1.0, 1.0, 1.0]), Material.DIELECTRIC, 1.5),
    
    # 粗い銅球（右奥）
    (np.array([8, 1.0, -20]), 1.8, np.array([0.95, 0.65, 0.3]), Material.METAL, 0.7),
    
    # 緑の拡散反射球（右手前）
    (np.array([6, 1.2, -16]), 1.5, np.array([0.2, 0.85, 0.3]), Material.DIFFUSE, 0.0),
    
    # 小さい金属球（左手前、補助的）
    (np.array([-4, 0.7, -15]), 0.8, np.array([0.85, 0.85, 0.9]), Material.METAL, 0.3),
]

# 複数の光源
light_sources = [
    (np.array([12, 25, 8]), np.array([1.0, 1.0, 0.98]), 1.2),    # 主光源（暖白色）
    (np.array([-15, 18, -5]), np.array([0.6, 0.75, 1.0]), 0.5),  # 補助光源（寒色）
]

# 環境光
ambient_light = np.array([0.2, 0.2, 0.25])

# ランダムな単位ベクトルを生成（球座標で高速化）
def random_in_unit_sphere():
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arccos(np.random.uniform(-1, 1))
    sin_phi = np.sin(phi)
    return np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), np.cos(phi)])

# Fresnel Schlick近似
def schlick_fresnel(cos_angle, ior):
    r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
    return r0 + (1 - r0) * (1 - cos_angle) ** 5

# レイと球の交差判定
def intersect(origin, direction):
    t_min = float('inf') 
    hit_obj = None
    for sphere in spheres:
        center, radius, _, _, _ = sphere
        oc = origin - center 
        b = np.dot(oc, direction)
        oc_sq = np.dot(oc, oc)
        c_val = oc_sq - radius * radius
        disc = b * b - c_val
        if disc > 0:
            sqrt_disc = math.sqrt(disc)
            t = -b - sqrt_disc
            if 0.001 < t < t_min:
                t_min = t
                hit_obj = sphere
    return t_min, hit_obj

# トレース
def trace(origin, direction, depth, rng_state=None):
    if depth <= 0:
        return np.array([0.0, 0.0, 0.0])
    
    t, obj = intersect(origin, direction)
    if not obj:
        # スカイボックス - グラデーション背景
        t = 0.5 * (direction[1] + 1.0)
        return np.array([0.2, 0.2, 0.3]) * (1 - t) + np.array([0.4, 0.6, 0.8]) * t

    center, radius, color, material, param = obj
    hit_pos = origin + direction * t
    
    # 法線の計算
    normal = hit_pos - center
    normal = normal / np.linalg.norm(normal)
    
    # 法線の向きを正す（背面除去）
    if np.dot(direction, normal) > 0:
        normal = -normal

    # 環境光の基本値
    result = ambient_light * color

    # 各光源からのライティング
    for light_pos, light_color, intensity in light_sources:
        light_dir = light_pos - hit_pos
        dist = np.linalg.norm(light_dir)
        light_dir = light_dir / dist
        
        # シャドウチェック
        t_shadow, shadow_obj = intersect(hit_pos + normal * 0.001, light_dir)
        
        if shadow_obj is None or t_shadow > dist:
            # ライティング計算
            cos_angle = max(0.0, np.dot(normal, light_dir))
            attenuation = intensity / (dist * dist * 0.01)
            
            if material == Material.DIFFUSE:
                # ランバートの余弦則
                result += color * light_color * cos_angle * attenuation
            
            elif material == Material.METAL:
                # 金属反射（Blinn-Phong的）
                half_vec = light_dir - direction
                half_len_sq = np.dot(half_vec, half_vec)
                if half_len_sq > 1e-12:
                    half_dir = half_vec / math.sqrt(half_len_sq)
                    spec_base = max(0.0, np.dot(normal, half_dir))
                    if spec_base > 1e-6:
                        spec = spec_base ** (1.0 / (param + 0.01))
                        result += light_color * spec * attenuation * 0.5
            
            elif material == Material.DIELECTRIC:
                # ガラスの表面反射
                result += light_color * cos_angle * attenuation * 0.1

    # 再帰的反射/屈折
    if material == Material.DIFFUSE:
        # ランダムな拡散方向
        random_dir = normal + random_in_unit_sphere()
        random_dir_len_sq = np.dot(random_dir, random_dir)
        if random_dir_len_sq > 1e-6:
            random_dir = random_dir / math.sqrt(random_dir_len_sq)
            reflected = trace(hit_pos, random_dir, depth - 1)
            result += color * reflected * 0.4
    
    elif material == Material.METAL:
        # 金属の鏡面反射（粗さで散乱）
        dot_prod = np.dot(direction, normal)
        ref_dir = direction - 2 * dot_prod * normal
        random_scatter = random_in_unit_sphere() * param * 0.3
        ref_dir = ref_dir + random_scatter
        ref_dir_len_sq = np.dot(ref_dir, ref_dir)
        if ref_dir_len_sq > 1e-6:
            ref_dir = ref_dir / math.sqrt(ref_dir_len_sq)
            reflected = trace(hit_pos, ref_dir, depth - 1)
            result += reflected * 0.8
    
    elif material == Material.DIELECTRIC:
        # ガラスの屈折と反射
        ior = param
        dot_prod = np.dot(direction, normal)
        cos_i = -dot_prod
        cos_i = np.clip(cos_i, -1.0, 1.0)  # 数値誤差を防ぐ
        
        cos_i_sq = cos_i * cos_i
        sin_i_sq = max(0.0, 1.0 - cos_i_sq)
        sin_i = math.sqrt(sin_i_sq)
        
        # 全反射判定
        sin_t = sin_i / ior
        if sin_t > 1.0:
            # 全反射
            ref_dir = direction - 2 * dot_prod * normal
            reflected = trace(hit_pos, ref_dir, depth - 1)
            result += reflected
        else:
            # Fresnel効果
            cos_t_sq = max(0.0, 1.0 - sin_t * sin_t)
            cos_t = math.sqrt(cos_t_sq)
            fresnel = schlick_fresnel(cos_i, ior)
            
            # 反射
            ref_dir = direction - 2 * dot_prod * normal
            reflected = trace(hit_pos, ref_dir, depth - 1)
            
            # 屈折
            refr_dir = (sin_i / ior) * (direction - cos_i * normal) - cos_t * normal
            refracted = trace(hit_pos, refr_dir, depth - 1)
            
            result += reflected * fresnel + refracted * (1.0 - fresnel) * 0.95

    return np.clip(result, 0, 1)

# レンダリングと画像の出力
W, H = 600, 450  # 解像度
num_samples = 6  # アンチエイリアシングのサンプル数（高速化で余裕ができた）
num_depth = 4    # トレース深度
print(f"レンダリングを実行中: 画像サイズ ({W}x{H}), サンプル: {num_samples}, 深度: {num_depth}")

camera_origin = np.array([2.0, 3.0, 8.0])  # カメラ位置（右高い）
camera_target = np.array([0.5, 1.5, -18.0])  # 注視点（ガラス球付近）

# カメラの方向ベクトルを計算
forward = camera_target - camera_origin
forward = forward / np.linalg.norm(forward)
right = np.cross(forward, np.array([0, 1, 0]))
right = right / np.linalg.norm(right)
up = np.cross(right, forward)
up = up / np.linalg.norm(up)

fov = 45.0  # 視野角
aspect_ratio = W / H
vfov_rad = math.radians(fov)
viewport_height = 2.0 * math.tan(vfov_rad / 2.0)
viewport_width = viewport_height * aspect_ratio

with open(f"render_depth_{num_depth}.ppm", "w") as f:
    f.write(f"P3\n{W} {H}\n255\n")
    
    for y in range(H):
        if y % 50 == 0:
            print(f"  レンダリング進行中: {y}/{H}")
        
        for x in range(W):
            pixel_color = np.array([0.0, 0.0, 0.0])
            
            # マルチサンプリング
            for s in range(num_samples):
                # サブピクセル位置
                u = (x + np.random.uniform(0, 1)) / W
                v = (y + np.random.uniform(0, 1)) / H
                
                # ビューポート上の位置
                viewport_x = (u - 0.5) * viewport_width
                viewport_y = -(v - 0.5) * viewport_height
                
                # レイ方向
                direction = forward + viewport_x * right + viewport_y * up
                direction = direction / np.linalg.norm(direction)
                
                # トレース
                sample_color = trace(camera_origin, direction, num_depth)
                pixel_color += sample_color
            
            # 平均化
            pixel_color = pixel_color / num_samples
            
            # ガンマ補正 (ガンマ = 2.2)
            pixel_color = np.power(pixel_color, 1.0 / 2.2)
            
            # 0-255に変換
            color_255 = np.clip(pixel_color, 0, 1) * 255
            r, g, b = color_255.astype(int)
            
            f.write(f"{r} {g} {b} ")
        f.write("\n")

print(f"完了！'render_depth_{num_depth}.ppm' に画像を保存しました。")