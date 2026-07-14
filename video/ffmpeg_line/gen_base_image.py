"""
gen_base_image.py — 程序生成深山夜雨靜景底圖
輸出: video/ffmpeg_line/assets/base.png  (1920x1080, RGB)

設計:
  - 天空漸層 (深藍→黑,帶微淡星光)
  - 遠山稜線(淡藍灰,3-4 層,各有不同高度)
  - 近山剪影(深藍黑)
  - 針葉樹剪影(最近景)
  - 薄霧帶 (中間高度,半透明白藍)

完全自產,零授權問題。使用 numpy + PIL。
"""

import pathlib
import numpy as np
from PIL import Image

W, H = 1920, 1080
OUT = pathlib.Path(__file__).parent / "assets" / "base.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(77)

# ---------------------------------------------------------
# 底層:天空漸層 (頂部深黑→中部午夜藍→底部略亮)
# ---------------------------------------------------------
img = np.zeros((H, W, 3), dtype=np.float32)

for y in range(H):
    t = y / H   # 0=top, 1=bottom
    # top: almost black (r=2,g=3,b=8)
    # middle sky: midnight blue (r=5,g=8,b=22)
    # horizon: slightly lighter blue-grey (r=10,g=14,b=28)
    r = np.interp(t, [0.0, 0.5, 1.0], [2, 5, 12]) / 255.0
    g = np.interp(t, [0.0, 0.5, 1.0], [3, 8, 16]) / 255.0
    b = np.interp(t, [0.0, 0.5, 1.0], [10, 22, 35]) / 255.0
    img[y, :, 0] = r
    img[y, :, 1] = g
    img[y, :, 2] = b

# 細微星光 (極少,非常暗)
n_stars = 120
sy = rng.integers(0, H // 2, n_stars)
sx = rng.integers(0, W, n_stars)
bright = rng.uniform(0.04, 0.12, n_stars)
for i in range(n_stars):
    b2 = bright[i]
    img[sy[i], sx[i], :] += b2
    if sy[i] > 0 and sy[i] < H - 1 and sx[i] > 0 and sx[i] < W - 1:
        img[sy[i]-1, sx[i], :] += b2 * 0.3
        img[sy[i]+1, sx[i], :] += b2 * 0.3
        img[sy[i], sx[i]-1, :] += b2 * 0.3
        img[sy[i], sx[i]+1, :] += b2 * 0.3


# ---------------------------------------------------------
# 輔助:用 perlin-like 1D 噪聲生稜線
# ---------------------------------------------------------
def gen_ridgeline(width: int, n_octaves: int, persistence: float,
                  seed_rng, smoothing: int = 8) -> np.ndarray:
    """回傳長度=width 的 0..1 高度陣列。"""
    line = np.zeros(width, dtype=np.float64)
    amp = 1.0
    freq = 1
    for _ in range(n_octaves):
        pts = seed_rng.uniform(0, 1, width // freq + 2)
        xs_in = np.linspace(0, len(pts) - 1, width)
        layer = np.interp(xs_in, np.arange(len(pts)), pts)
        line += layer * amp
        amp *= persistence
        freq *= 2
    line -= line.min()
    line /= (line.max() + 1e-9)
    # 平滑
    from scipy.ndimage import uniform_filter1d
    line = uniform_filter1d(line, size=smoothing)
    line -= line.min()
    line /= (line.max() + 1e-9)
    return line


# ---------------------------------------------------------
# 遠山層 (3 層,由遠到近,顏色漸深)
# ---------------------------------------------------------
mountain_layers = [
    # (base_y_frac, height_frac, color_rgb_255, octaves, persistence, smooth)
    (0.38, 0.18, (8, 14, 30),  4, 0.55, 30),   # 最遠,最淡
    (0.44, 0.20, (6, 10, 22),  5, 0.58, 20),   # 中遠
    (0.52, 0.24, (4,  7, 18),  5, 0.60, 15),   # 中近
]

for (base_y, h_frac, col, noct, pers, sm) in mountain_layers:
    ridge = gen_ridgeline(W, noct, pers, rng, sm)
    ridge_y = (base_y + h_frac * (1.0 - ridge)) * H  # 越高 ridge 值 → 越靠近頂部
    ridge_y = ridge_y.astype(int).clip(0, H - 1)

    cr, cg, cb = [c / 255.0 for c in col]
    for x in range(W):
        y_top = ridge_y[x]
        # 填色到畫面底(或下一層蓋掉)
        img[y_top:, x, 0] = cr
        img[y_top:, x, 1] = cg
        img[y_top:, x, 2] = cb


# ---------------------------------------------------------
# 近景山體 (更高更深,強調層次)
# ---------------------------------------------------------
near_ridge = gen_ridgeline(W, 6, 0.62, rng, 10)
near_base_y = 0.62
near_h_frac = 0.22
near_ridge_y = ((near_base_y + near_h_frac * (1.0 - near_ridge)) * H).astype(int).clip(0, H - 1)
near_col = (3, 5, 12)
cr, cg, cb = [c / 255.0 for c in near_col]
for x in range(W):
    y_top = near_ridge_y[x]
    img[y_top:, x, 0] = cr
    img[y_top:, x, 1] = cg
    img[y_top:, x, 2] = cb


# ---------------------------------------------------------
# 針葉樹剪影 (最近景,沿底部鋸齒狀)
# ---------------------------------------------------------
def draw_conifer(img_arr, cx, base_y, tree_h, tree_w, color):
    """在 img_arr 上繪製單棵針葉樹剪影(多層三角形)。"""
    cr, cg, cb = [c / 255.0 for c in color]
    n_tiers = max(2, int(tree_h / 18))
    tier_h = tree_h / n_tiers
    for tier in range(n_tiers):
        t_frac = tier / n_tiers
        w_half = int(tree_w * 0.5 * (1.0 - t_frac * 0.3))
        h_start = int(base_y - tree_h + tier * tier_h)
        h_end = int(base_y - tree_h + (tier + 1) * tier_h) + 1
        for y in range(max(0, h_start), min(img_arr.shape[0], h_end)):
            y_frac = (y - h_start) / max(tier_h, 1)
            half = int(w_half * y_frac)
            x0 = max(0, cx - half)
            x1 = min(img_arr.shape[1], cx + half + 1)
            img_arr[y, x0:x1, 0] = cr
            img_arr[y, x0:x1, 1] = cg
            img_arr[y, x0:x1, 2] = cb

tree_col = (2, 3, 8)
# 生成一排近景樹
n_trees = 35
tree_xs = rng.integers(20, W - 20, n_trees)
tree_hs = rng.integers(60, 140, n_trees)
tree_ws = (tree_hs * rng.uniform(0.25, 0.40, n_trees)).astype(int)
# 底部 y 在畫面 75-88% 高
base_ys = rng.integers(int(H * 0.75), int(H * 0.88), n_trees)

for i in range(n_trees):
    draw_conifer(img, int(tree_xs[i]), int(base_ys[i]),
                 int(tree_hs[i]), int(tree_ws[i]), tree_col)

# 額外第二排樹(較小,稍遠)
n_trees2 = 25
tree_xs2 = rng.integers(0, W, n_trees2)
tree_hs2 = rng.integers(35, 75, n_trees2)
tree_ws2 = (tree_hs2 * rng.uniform(0.22, 0.35, n_trees2)).astype(int)
base_ys2 = rng.integers(int(H * 0.68), int(H * 0.76), n_trees2)
tree_col2 = (3, 4, 10)
for i in range(n_trees2):
    draw_conifer(img, int(tree_xs2[i]), int(base_ys2[i]),
                 int(tree_hs2[i]), int(tree_ws2[i]), tree_col2)


# ---------------------------------------------------------
# 薄霧漸層帶 (中段,y=55%~75%)
# ---------------------------------------------------------
mist_top = int(H * 0.52)
mist_bot = int(H * 0.72)
for y in range(mist_top, mist_bot):
    # 鐘形強度
    t = (y - mist_top) / (mist_bot - mist_top)
    alpha = 0.07 * np.sin(t * np.pi) ** 1.5
    img[y, :, 0] = np.clip(img[y, :, 0] + alpha * 0.6, 0, 1)
    img[y, :, 1] = np.clip(img[y, :, 1] + alpha * 0.7, 0, 1)
    img[y, :, 2] = np.clip(img[y, :, 2] + alpha * 1.0, 0, 1)


# ---------------------------------------------------------
# 輸出
# ---------------------------------------------------------
img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
Image.fromarray(img_uint8).save(str(OUT))
print(f"[gen_base_image] saved {OUT}  ({W}x{H})")
