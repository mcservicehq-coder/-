"""画像（餅つき姿の男性）を参照し、プリミティブを組み合わせて3Dモデル(GLB/OBJ)を生成する。

座標系: Y上向き、正面 +Z、単位メートル（身長 約1.7m）。
キャラクターの右手 = -X 側（画像の向かって左）。
"""
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix

SKIN = [232, 178, 140, 255]
SKIN_DARK = [205, 145, 110, 255]
WHITE = [245, 245, 245, 255]
RED = [220, 30, 35, 255]
NAVY = [25, 30, 60, 255]
WOOD = [214, 160, 105, 255]
WOOD_DARK = [160, 105, 60, 255]
MOUTH = [110, 25, 30, 255]
TONGUE = [200, 80, 90, 255]
HAIR = [50, 35, 30, 255]

parts = {}


def colored(mesh, color):
    mesh.merge_vertices()
    mesh.visual.vertex_colors = color
    return mesh


def add(name, mesh, color):
    parts[name] = colored(mesh, color)


def ellipsoid(center, radii, subdiv=3):
    m = trimesh.creation.icosphere(subdivisions=subdiv)
    m.apply_scale(radii)
    m.apply_translation(center)
    return m


def align_z_to(vec):
    """Z軸を vec 方向へ向ける回転行列。"""
    z = np.array([0, 0, 1.0])
    v = np.asarray(vec, float)
    v = v / np.linalg.norm(v)
    axis = np.cross(z, v)
    s = np.linalg.norm(axis)
    if s < 1e-8:
        return np.eye(4) if v[2] > 0 else rotation_matrix(np.pi, [1, 0, 0])
    return rotation_matrix(np.arctan2(s, np.dot(z, v)), axis / s)


def limb(p0, p1, r0, r1=None, sections=24):
    """p0→p1 の先細り円柱＋両端球（関節）。"""
    r1 = r0 if r1 is None else r1
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    length = np.linalg.norm(p1 - p0)
    m = trimesh.creation.cylinder(radius=1, height=1, sections=sections)
    v = m.vertices.copy()
    t = v[:, 2] + 0.5
    rad = r0 + (r1 - r0) * t
    v[:, 0] *= rad
    v[:, 1] *= rad
    v[:, 2] = t * length
    m.vertices = v
    m.apply_transform(align_z_to(p1 - p0))
    m.apply_translation(p0)
    j0 = trimesh.creation.icosphere(subdivisions=2, radius=r0)
    j0.apply_translation(p0)
    j1 = trimesh.creation.icosphere(subdivisions=2, radius=r1)
    j1.apply_translation(p1)
    return trimesh.util.concatenate([m, j0, j1])


def cylinder_between(p0, p1, r, sections=32):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    m = trimesh.creation.cylinder(radius=r, height=np.linalg.norm(p1 - p0), sections=sections)
    m.apply_transform(align_z_to(p1 - p0))
    m.apply_translation((p0 + p1) / 2)
    return m


# ---------------- 胴体 ----------------
WAIST_Y = 0.98
add("chest", ellipsoid([0, 1.32, 0.0], [0.21, 0.20, 0.14]), SKIN)
add("shoulders", ellipsoid([0, 1.40, -0.01], [0.25, 0.08, 0.12]), SKIN)
add("pecs", trimesh.util.concatenate([
    ellipsoid([-0.09, 1.31, 0.10], [0.10, 0.07, 0.05]),
    ellipsoid([0.09, 1.31, 0.10], [0.10, 0.07, 0.05])]), SKIN)
add("nipples", trimesh.util.concatenate([
    ellipsoid([-0.10, 1.29, 0.148], [0.012, 0.010, 0.006], 1),
    ellipsoid([0.10, 1.29, 0.148], [0.012, 0.010, 0.006], 1)]), SKIN_DARK)
add("chest_hair", ellipsoid([0, 1.30, 0.128], [0.10, 0.07, 0.025]), HAIR)
add("neck", cylinder_between([0, 1.40, 0], [0, 1.52, 0.01], 0.075), SKIN)

# 腹巻（赤白ストライプ）: 楕円筒の側面を角度で塗り分け
hara = trimesh.creation.cylinder(radius=1, height=0.30, sections=96)
hara.apply_scale([0.215, 0.165, 1])
hara.apply_transform(rotation_matrix(-np.pi / 2, [1, 0, 0]))
hara.apply_translation([0, 1.10, 0.005])
hara.unmerge_vertices()
c = hara.triangles_center
ang = np.arctan2(c[:, 0], c[:, 2])
stripe = (np.floor((ang + np.pi) / (2 * np.pi) * 14) % 2).astype(int)
cols = np.where(stripe[:, None] == 0, RED, WHITE)
cap = np.abs(hara.face_normals[:, 1]) > 0.9
cols[cap] = RED
hara.visual.face_colors = cols
parts["haramaki"] = hara

# パンツ（白ステテコ風）
add("shorts_hip", ellipsoid([0, 0.93, 0.0], [0.22, 0.13, 0.15]), WHITE)

# ---------------- 脚（がに股・膝曲げ） ----------------
for s, name in [(-1, "R"), (1, "L")]:
    hip = [s * 0.11, 0.90, 0.0]
    knee = [s * 0.25, 0.52, 0.10]
    ankle = [s * 0.27, 0.10, 0.02]
    add(f"thigh_short_{name}", limb(hip, [s * 0.20, 0.66, 0.07], 0.12, 0.105), WHITE)
    add(f"thigh_{name}", limb([s * 0.20, 0.66, 0.07], knee, 0.09, 0.075), SKIN)
    add(f"shin_{name}", limb(knee, ankle, 0.072, 0.05), SKIN)
    foot = ellipsoid([s * 0.28, 0.04, 0.07], [0.055, 0.04, 0.12])
    add(f"foot_{name}", foot, SKIN)

# ---------------- 頭部 ----------------
HEAD = np.array([0.0, 1.66, 0.03])
add("head", ellipsoid(HEAD, [0.105, 0.13, 0.115]), SKIN)
add("jaw", ellipsoid(HEAD + [0, -0.06, 0.02], [0.095, 0.08, 0.095]), SKIN)
add("ears", trimesh.util.concatenate([
    ellipsoid(HEAD + [-0.105, -0.01, 0], [0.02, 0.035, 0.025], 2),
    ellipsoid(HEAD + [0.105, -0.01, 0], [0.02, 0.035, 0.025], 2)]), SKIN)
add("nose", ellipsoid(HEAD + [0, -0.015, 0.118], [0.022, 0.025, 0.02], 2), SKIN)
add("mouth", ellipsoid(HEAD + [0, -0.075, 0.098], [0.042, 0.032, 0.02], 2), MOUTH)
add("tongue", ellipsoid(HEAD + [0, -0.088, 0.105], [0.026, 0.012, 0.012], 2), TONGUE)
add("teeth", ellipsoid(HEAD + [0, -0.052, 0.108], [0.032, 0.007, 0.008], 2), WHITE)
eyes = []
brows = []
for s in (-1, 1):
    eyes.append(ellipsoid(HEAD + [s * 0.04, 0.02, 0.103], [0.014, 0.009, 0.006], 2))
    b = ellipsoid(HEAD + [s * 0.042, 0.045, 0.104], [0.024, 0.006, 0.006], 2)
    brows.append(b)
add("eyes", trimesh.util.concatenate(eyes), HAIR)
add("brows", trimesh.util.concatenate(brows), HAIR)

# 鉢巻（白地に紺の水玉）＋結び目
band = trimesh.creation.torus(major_radius=1, minor_radius=0.02, major_sections=96, minor_sections=12)
band.apply_scale([0.112, 0.115, 1.0])
band.apply_scale([1, 1, 1.4])  # 縦幅
band.apply_transform(rotation_matrix(np.pi / 2 + 0.12, [1, 0, 0]))
band.apply_translation(HEAD + [0, 0.07, 0.0])
band.unmerge_vertices()
bc = band.triangles_center - (HEAD + [0, 0.07, 0])
a = np.arctan2(bc[:, 0], bc[:, 2])
dots = (np.sin(a * 22) > 0.6) & (np.abs(bc[:, 1]) < 0.012)
band.visual.face_colors = np.where(dots[:, None], NAVY, WHITE)
parts["hachimaki"] = band
knot = HEAD + [-0.10, 0.08, 0.03]  # 画像では向かって左（キャラの右）側に結び目
add("hachimaki_knot", trimesh.util.concatenate([
    ellipsoid(knot, [0.025, 0.025, 0.025], 2),
    ellipsoid(knot + [-0.03, 0.025, 0.01], [0.035, 0.018, 0.012], 2),
    ellipsoid(knot + [-0.03, -0.02, 0.01], [0.03, 0.016, 0.012], 2)]), WHITE)

# ---------------- 腕と杵 ----------------
# 杵：柄は体の前を斜めに横切り、頭（臼をつく側）はキャラの右下（-X）
grip_R = np.array([-0.22, 1.02, 0.30])   # 右手（柄の中ほど）
grip_L = np.array([0.30, 0.86, 0.22])    # 左手（柄尻付近）
shaft_dir = grip_R - grip_L
shaft_dir /= np.linalg.norm(shaft_dir)
shaft_end = grip_L - shaft_dir * 0.06
shaft_tip = grip_R + shaft_dir * 0.42
add("kine_shaft", cylinder_between(shaft_end, shaft_tip, 0.018), WOOD)
# 杵頭（柄と直交する円柱）
head_axis = np.cross(shaft_dir, [0, 0, 1.0])
head_axis /= np.linalg.norm(head_axis)
add("kine_head", cylinder_between(shaft_tip - head_axis * 0.17, shaft_tip + head_axis * 0.17, 0.065, 40), WOOD)
add("kine_ends", trimesh.util.concatenate([
    cylinder_between(shaft_tip - head_axis * 0.172, shaft_tip - head_axis * 0.165, 0.066, 40),
    cylinder_between(shaft_tip + head_axis * 0.165, shaft_tip + head_axis * 0.172, 0.066, 40)]), WOOD_DARK)

for s, name, grip in [(-1, "R", grip_R), (1, "L", grip_L)]:
    shoulder = np.array([s * 0.24, 1.40, -0.01])
    if name == "R":
        elbow = np.array([-0.32, 1.12, 0.12])
    else:
        elbow = np.array([0.36, 1.06, 0.02])
    add(f"upper_arm_{name}", limb(shoulder, elbow, 0.07, 0.058), SKIN)
    add(f"forearm_{name}", limb(elbow, grip - [0, 0, 0.0], 0.055, 0.045), SKIN)
    add(f"hand_{name}", ellipsoid(grip, [0.05, 0.045, 0.045], 2), SKIN)

# ---------------- 臼（手前に置く） ----------------
usu = trimesh.creation.cylinder(radius=0.28, height=0.45, sections=64)
usu.apply_transform(rotation_matrix(-np.pi / 2, [1, 0, 0]))
usu.apply_translation([-0.55, 0.225, 0.55])
add("usu", usu, WOOD)
bowl = ellipsoid([-0.55, 0.45, 0.55], [0.22, 0.02, 0.22])
add("usu_inner", bowl, WOOD_DARK)
add("mochi", ellipsoid([-0.55, 0.465, 0.55], [0.15, 0.04, 0.15]), WHITE)

# ---------------- 出力 ----------------
scene = trimesh.Scene()
for k, m in parts.items():
    scene.add_geometry(m, node_name=k, geom_name=k)

scene.export("mochitsuki_man.glb", include_normals=True)
trimesh.util.concatenate(list(parts.values())).export("mochitsuki_man.obj")
b = scene.bounds
print("parts:", len(parts), "bounds:", b.round(3).tolist(), "size:", (b[1] - b[0]).round(3).tolist())
