import numpy as np
from PIL import Image
import os

W, H = 1440, 2560
OUT = "."

def save(arr, name):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(os.path.join(OUT, f"{name}.png"))
    print(f"  {name}.png  {arr.shape[1]}x{arr.shape[0]}")

# --- noise helpers ---

def perlin_noise(w, h, scale, octaves=4, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), dtype=np.float64)
    amp = 1.0
    freq = 1.0
    max_val = 0.0
    for _ in range(octaves):
        sw = max(2, int(w / (scale * freq)))
        sh = max(2, int(h / (scale * freq)))
        grid = rng.uniform(-1, 1, (sh + 1, sw + 1))
        xs = np.linspace(0, sw, w, endpoint=False)
        ys = np.linspace(0, sh, h, endpoint=False)
        fx, fy = np.floor(xs).astype(int), np.floor(ys).astype(int)
        dx, dy = xs - fx, ys - fy
        sx, sy = dx * dx * (3 - 2 * dx), dy * dy * (3 - 2 * dy)
        fxx, fyy = np.meshgrid(fx, fy)
        dxx, dyy = np.meshgrid(dx, dy)
        sxx, syy = np.meshgrid(sx, sy)
        for j in range(2):
            for i in range(2):
                g = grid[fyy + j, fxx + i]
                wgt = (1 - i + (2 * i - 1) * sxx) * (1 - j + (2 * j - 1) * syy)
                out += amp * g * wgt
        amp *= 0.5
        freq *= 2.0
        max_val += amp
    return np.clip(out / max_val, -1, 1)

def tileable_noise(w, h, scale, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), dtype=np.float64)
    gw, gh = w // scale + 2, h // scale + 2
    g = rng.uniform(-1, 1, (gh, gw))
    for k in range(4):
        xs = np.linspace(0, w / scale, w, endpoint=False) + k * 0.618
        ys = np.linspace(0, h / scale, h, endpoint=False) + k * 1.236
        fx, fy = np.floor(xs).astype(int) % gw, np.floor(ys).astype(int) % gh
        dx, dy = xs - np.floor(xs), ys - np.floor(ys)
        sx, sy = dx * dx * (3 - 2 * dx), dy * dy * (3 - 2 * dy)
        fxx, fyy = np.meshgrid(fx, fy)
        dxx, dyy = np.meshgrid(dx, dy)
        sxx, syy = np.meshgrid(sx, sy)
        for dj in range(2):
            for di in range(2):
                gi = g[(fyy + dj) % gh, (fxx + di) % gw]
                wgt = (1 - di + (2 * di - 1) * sxx) * (1 - dj + (2 * dj - 1) * syy)
                out += gi * wgt
    return out / np.std(out)

def voronoi(w, h, n_cells, seed=0, jitter=0.5):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, (n_cells, 2)) * np.array([[w, h]])
    yy, xx = np.mgrid[0:h, 0:w]
    dists = np.full((h, w), np.inf)
    nearest = np.zeros((h, w), dtype=int)
    for i, (cx, cy) in enumerate(pts):
        d = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = d < dists
        dists[mask] = d[mask]
        nearest[mask] = i
    cell_vals = rng.uniform(0, 1, n_cells)[nearest]
    dist_map = np.sqrt(dists) / np.sqrt(w * h / n_cells) * 2
    return cell_vals, dist_map, pts

# ===== PHASE 1 — quiet monochrome =====

def p1_perlin():
    n = perlin_noise(W, H, scale=60, seed=1)
    img = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    save(img, "phase1-overlay-4-perlin")

def p1_voronoi():
    cells, dist, pts = voronoi(W, H, 120, seed=2)
    img = (cells * 200 + 28).astype(np.uint8)
    save(img, "phase1-overlay-5-voronoi")

def p1_multilayer():
    n = perlin_noise(W, H, scale=80, seed=3)
    cells, dist, pts = voronoi(W, H, 80, seed=4)
    scan = np.fromfunction(lambda y, x: (y % 3 == 0).astype(float), (H, W))
    blend = n * 0.4 + (cells * 2 - 1) * 0.3 + scan * 0.3
    img = ((blend * 0.5 + 0.5) * 230 + 12).astype(np.uint8)
    save(img, "phase1-overlay-6-multilayer")

def p1_tiled():
    n = tileable_noise(W, H, scale=80, seed=5)
    img = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    save(img, "phase1-overlay-7-tiled")

def p1_psychedelic():
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W / 2)
    a = np.arctan2(yy - H / 2, xx - W / 2)
    n1 = perlin_noise(W, H, scale=40, seed=6)
    val = np.sin(r * 30 + n1 * 4) * np.cos(a * 6 + n1 * 3)
    img = ((val * 0.5 + 0.5) * 255).astype(np.uint8)
    save(img, "phase1-overlay-8-psychedelic")

# ===== PHASE 2 — active glitch/tear/noise =====

def p2_perlin():
    n = perlin_noise(W, H, scale=25, seed=7, octaves=6)
    img = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    save(img, "phase2-overlay-4-perlin")

def p2_voronoi():
    cells, dist, pts = voronoi(W, H, 200, seed=8)
    rng = np.random.default_rng(8)
    cell_edges = dist < 0.08
    offsets = (cells * 255).astype(np.uint8)
    offsets[cell_edges] = 255 - offsets[cell_edges]
    save(offsets, "phase2-overlay-5-voronoi")

def p2_multilayer():
    n = perlin_noise(W, H, scale=30, seed=9, octaves=5)
    cells, dist, pts = voronoi(W, H, 150, seed=10)
    rng = np.random.default_rng(11)
    glitch_lines = np.zeros((H, W))
    for _ in range(60):
        y = rng.integers(0, H)
        hh = rng.integers(1, 8)
        shift = rng.integers(-40, 40)
        glitch_lines[y:y + hh] = shift * 0.01
    blend = n * 0.35 + (cells * 2 - 1) * 0.25 + glitch_lines * 2 + dist * 0.15
    img = ((blend * 0.5 + 0.5) * 255).astype(np.uint8)
    save(img, "phase2-overlay-6-multilayer")

def p2_tiled():
    n = tileable_noise(W, H, scale=30, seed=12)
    edges = np.abs(np.diff(n, axis=1, append=n[:, :1])) > 1.5
    n[edges] = 1.0
    n[~edges] = (n[~edges] * 0.5 + 0.5)
    img = (n * 255).astype(np.uint8)
    save(img, "phase2-overlay-7-tiled")

def p2_psychedelic():
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W / 2)
    a = np.arctan2(yy - H / 2, xx - W / 2)
    n1 = perlin_noise(W, H, scale=20, seed=13, octaves=5)
    n2 = perlin_noise(W, H, scale=50, seed=14)
    val = np.sin(r * 50 + n1 * 8) * np.sin(a * 10 + n2 * 5) * np.cos(r * 15 + n2 * 3)
    val += np.sin(xx * 0.08 + yy * 0.12) * 0.3
    img = ((val * 0.5 + 0.5) * 255).astype(np.uint8)
    save(img, "phase2-overlay-8-psychedelic")

# ===== RUN =====

print("Phase 1 (quiet monochrome):")
p1_perlin()
p1_voronoi()
p1_multilayer()
p1_tiled()
p1_psychedelic()

print("Phase 2 (active):")
p2_perlin()
p2_voronoi()
p2_multilayer()
p2_tiled()
p2_psychedelic()

print("Done — 10 textures generated")
