import math
import numpy as np

# シーン定義
# 球体: (中心座標, 半径, 色, 反射率)
spheres = [
    (np.array([0, -10004, -20]), 10000, np.array([0.2, 0.2, 0.2]), 0.1), # 地面 (デカい球として表現)
    (np.array([0, 0, -20]), 4, np.array([1.0, 0.3, 0.3]), 0.5),          # 赤い球
    (np.array([5, -1, -15]), 2, np.array([0.3, 0.3, 1.0]), 0.4),         # 青い球
    (np.array([-5, 0, -15]), 3, np.array([0.3, 1.0, 0.3]), 0.4)          # 緑の球
]
light_pos = np.array([10, 20, 10]) # 光源座標

# レイと球の交差判定
def intersect(origin, direction):
    t_min = float('inf') 
    hit_obj = None
    for sphere in spheres:
        center, radius, _, _ = sphere
        oc = origin - center 
        b = np.dot(oc, direction) # 内積
        c_val = np.dot(oc, oc) - radius**2
        disc = b**2 - c_val
        if disc > 0:
            t = -b - math.sqrt(disc)
            if 0.001 < t < t_min:
                t_min = t
                hit_obj = sphere
    return t_min, hit_obj

# トレース
def trace(origin, direction, depth):
    if depth <= 0: return np.array([0.0, 0.0, 0.0]) # 再帰の深さ制限に到達したら黒
    
    t, obj = intersect(origin, direction)
    if not obj: return np.array([0.05, 0.05, 0.05]) # 背景色

    center, radius, color, reflectivity = obj
    hit_pos = origin + direction * t
    
    # 法線の計算と正規化
    normal = hit_pos - center
    normal = normal / np.linalg.norm(normal)

    # 光源へのレイ（シャドウレイ）
    light_dir = light_pos - hit_pos
    light_dir = light_dir / np.linalg.norm(light_dir)
    _, shadow_obj = intersect(hit_pos, light_dir)
    
    intensity = 0.0
    if shadow_obj is None:
        intensity = max(0.0, np.dot(normal, light_dir))

    # 拡散反射
    diffuse = color * intensity * (1 - reflectivity)

    # 鏡面反射
    # 反射ベクトル = 入射ベクトル - 2 * (入射・法線の内積) * 法線
    ref_dir = direction - 2 * np.dot(direction, normal) * normal
    reflected = trace(hit_pos, ref_dir, depth - 1)
    reflected = reflected * reflectivity

    return diffuse + reflected

# レンダリングと画像の出力
W, H = 400, 300 # 解像度
print(f"レンダリングを実行中: 画像サイズ ({W}x{H})")

camera_origin = np.array([0.0, 0.0, 0.0])
num_depth = 5

with open(f"render_depth_{num_depth}.ppm", "w") as f:
    f.write(f"P3\n{W} {H}\n255\n")
    for y in range(H):
        for x in range(W):
            dir_x = (x - W / 2) / W
            dir_y = -(y - H / 2) / W
            
            # レイの方向ベクトル (正規化しとく)
            direction = np.array([dir_x, dir_y, -1.0])
            direction = direction / np.linalg.norm(direction)
            
            # 光線を追跡
            pixel_color = trace(camera_origin, direction, num_depth)
            
            # NumPyのclipを使って0.0〜1.0の範囲に収め、255を掛けて整数化
            color_255 = np.clip(pixel_color, 0, 1) * 255
            r, g, b = color_255.astype(int)
            
            f.write(f"{r} {g} {b} ")
        f.write("\n")

print(f"完了 'render_depth_{num_depth}.ppm' に画像を保存しました。")