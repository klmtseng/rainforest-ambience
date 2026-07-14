"""
scene_rain.py — headless Blender 4.0.x bpy script
深山雨夜 10s loop (260 frames at 24fps, ffmpeg crossfade → 240-frame MP4)

Engine: Cycles CPU, 16 samples, no denoising
Resolution: 960x540
Rain loop closure: particle lifetime = 240 frames → frame 260 state ≈ frame 20
(good enough for 1s crossfade to close the loop)

Usage:
    blender -b --factory-startup -noaudio -E CYCLES \
        -P scene_rain.py -- \
        --frames 260 \
        --outdir /path/to/frames/
"""

import bpy
import math
import sys
import os
import random

# ── argument parsing ─────────────────────────────────────────────────────────
argv = sys.argv
try:
    sep = argv.index("--")
    args = argv[sep + 1:]
except ValueError:
    args = []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=260)
parser.add_argument("--outdir", type=str, default="/tmp/rainforest_frames/")
parser.add_argument("--samples", type=int, default=8)
pargs = parser.parse_args(args)

TOTAL_FRAMES = pargs.frames
OUTDIR = pargs.outdir
SAMPLES = pargs.samples
os.makedirs(OUTDIR, exist_ok=True)

# ── scene reset ───────────────────────────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
# clear all materials
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat)

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = TOTAL_FRAMES

# ── render settings ───────────────────────────────────────────────────────────
scene.render.engine = 'CYCLES'
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = False
scene.cycles.device = 'CPU'
scene.render.resolution_x = 960
scene.render.resolution_y = 540
scene.render.fps = 24
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.compression = 9

# ── world: cold blue night sky gradient ──────────────────────────────────────
world = bpy.data.worlds.new("NightSky")
scene.world = world
world.use_nodes = True
nt = world.node_tree
nt.nodes.clear()

bg = nt.nodes.new("ShaderNodeBackground")
grad = nt.nodes.new("ShaderNodeTexGradient")
tc = nt.nodes.new("ShaderNodeTexCoord")
map_node = nt.nodes.new("ShaderNodeMapping")
mix = nt.nodes.new("ShaderNodeMixRGB")
col1 = nt.nodes.new("ShaderNodeRGB")
col2 = nt.nodes.new("ShaderNodeRGB")
out = nt.nodes.new("ShaderNodeOutputWorld")

# Very dark blue-black at top, slightly lighter blue-grey at horizon
col1.outputs[0].default_value = (0.003, 0.005, 0.020, 1.0)   # deep night blue
col2.outputs[0].default_value = (0.010, 0.015, 0.040, 1.0)   # horizon blue-grey

# Gradient from bottom (horizon=lighter) to top (darker)
# Use object Y axis mapped to gradient
tc.location = (-800, 0)
map_node.location = (-600, 0)
map_node.inputs['Scale'].default_value = (1.0, 1.0, 1.0)
grad.location = (-400, 0)
grad.gradient_type = 'LINEAR'
mix.location = (-200, 0)
mix.blend_type = 'MIX'
col1.location = (-400, -200)
col2.location = (-400, -400)
bg.location = (0, 0)
out.location = (200, 0)

nt.links.new(tc.outputs['Generated'], map_node.inputs['Vector'])
nt.links.new(map_node.outputs['Vector'], grad.inputs['Vector'])
nt.links.new(grad.outputs['Fac'], mix.inputs[0])
nt.links.new(col2.outputs[0], mix.inputs[1])   # fac=0 → horizon
nt.links.new(col1.outputs[0], mix.inputs[2])   # fac=1 → zenith
nt.links.new(mix.outputs[0], bg.inputs['Color'])
bg.inputs['Strength'].default_value = 1.0
nt.links.new(bg.outputs['Background'], out.inputs['Surface'])

# ── helper: make a material ───────────────────────────────────────────────────
def make_emission_mat(name, color, strength=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs['Color'].default_value = (*color, 1.0)
    em.inputs['Strength'].default_value = strength
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs['Emission'], out_n.inputs['Surface'])
    return mat

def make_diffuse_mat(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    diff.inputs['Color'].default_value = (*color, 1.0)
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(diff.outputs['BSDF'], out_n.inputs['Surface'])
    return mat

# ── mountain ridges (3 layers, silhouettes) ───────────────────────────────────
# Each ridge is a subdivided plane distorted via vertex positions
# Dark near-black colors, slightly blue-tinted

def add_mountain_ridge(name, z_base, z_height_scale, y_pos, color, seed_val):
    """Create a jagged ridge plane as mountain silhouette."""
    rng = random.Random(seed_val)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, y_pos, 0))
    obj = bpy.context.active_object
    obj.name = name

    # Scale to cover full camera width + depth
    obj.scale = (14.0, 1.0, 1.0)
    bpy.ops.object.transform_apply(scale=True)

    # Switch to edit mode to subdivide
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    # subdivide multiple times for more vertices
    bpy.ops.mesh.subdivide(number_cuts=20)
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh = obj.data
    # Move vertices to create mountain profile
    for v in mesh.vertices:
        x = v.co.x
        # Multi-octave noise via sum of sines with random phase/freq
        h = 0.0
        for octave in range(6):
            freq = (2 ** octave) * 0.4
            phase = rng.uniform(0, math.pi * 2)
            amp = z_height_scale / (2 ** octave)
            h += amp * math.sin(x * freq + phase)
        # Ensure floor stays at z_base
        v.co.z = z_base + max(0.0, h)

    # Set material
    mat = make_diffuse_mat(f"{name}_mat", color)
    obj.data.materials.append(mat)

    # Disable shadow casting to save render time
    obj.cycles.is_shadow_catcher = False
    return obj

# Far ridge (lightest, smallest, furthest)
add_mountain_ridge("ridge_far", z_base=-1.0, z_height_scale=2.5,
                   y_pos=6.0, color=(0.008, 0.010, 0.022), seed_val=42)
# Mid ridge
add_mountain_ridge("ridge_mid", z_base=-1.5, z_height_scale=3.5,
                   y_pos=3.5, color=(0.004, 0.006, 0.015), seed_val=77)
# Near ridge (darkest, tallest, closest)
add_mountain_ridge("ridge_near", z_base=-2.0, z_height_scale=4.5,
                   y_pos=1.5, color=(0.002, 0.003, 0.008), seed_val=13)

# ── conifer tree clusters (cone geometry) ─────────────────────────────────────
def add_cone_tree(x, y, z_base, height, radius, color):
    bpy.ops.mesh.primitive_cone_add(
        vertices=8,
        radius1=radius,
        radius2=0.02,
        depth=height,
        location=(x, y, z_base + height / 2)
    )
    obj = bpy.context.active_object
    mat = make_diffuse_mat("tree_mat", color)
    obj.data.materials.append(mat)
    return obj

# Dark silhouette trees, near ridge
tree_color = (0.001, 0.002, 0.005)
rng_trees = random.Random(99)
for i in range(12):
    tx = rng_trees.uniform(-6.5, 6.5)
    th = rng_trees.uniform(1.2, 2.8)
    tr = rng_trees.uniform(0.3, 0.7)
    add_cone_tree(tx, y=1.8, z_base=-2.0 + rng_trees.uniform(-0.3, 0.3),
                  height=th, radius=tr, color=tree_color)

# Second row slightly further
for i in range(8):
    tx = rng_trees.uniform(-5.0, 5.0)
    th = rng_trees.uniform(1.0, 2.2)
    tr = rng_trees.uniform(0.25, 0.55)
    add_cone_tree(tx, y=2.8, z_base=-1.8 + rng_trees.uniform(-0.2, 0.2),
                  height=th, radius=tr, color=tree_color)

# ── volumetric fog (large cube with Principled Volume) ────────────────────────
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 4, 0))
fog_cube = bpy.context.active_object
fog_cube.name = "fog_volume"
fog_cube.scale = (20.0, 14.0, 6.0)
bpy.ops.object.transform_apply(scale=True)

fog_mat = bpy.data.materials.new("fog_mat")
fog_mat.use_nodes = True
nt_fog = fog_mat.node_tree
nt_fog.nodes.clear()
vol = nt_fog.nodes.new("ShaderNodeVolumePrincipled")
out_f = nt_fog.nodes.new("ShaderNodeOutputMaterial")
# Density ~0.02 for subtle atmospheric fog
vol.inputs['Density'].default_value = 0.02
vol.inputs['Color'].default_value = (0.6, 0.7, 1.0, 1.0)  # cool blue-white fog
nt_fog.links.new(vol.outputs['Volume'], out_f.inputs['Volume'])
fog_cube.data.materials.append(fog_mat)
# Make fog transparent in viewport but render it
fog_cube.display_type = 'WIRE'

# ── rain particle system ──────────────────────────────────────────────────────
# Emitter: large plane above the scene
bpy.ops.mesh.primitive_plane_add(size=18.0, location=(0, 4, 6))
rain_emitter = bpy.context.active_object
rain_emitter.name = "rain_emitter"

# Thin streak object (elongated cube) as rain drop instance
bpy.ops.mesh.primitive_cube_add(size=1, location=(100, 0, 0))  # off-screen
raindrop = bpy.context.active_object
raindrop.name = "raindrop_instance"
raindrop.scale = (0.005, 0.005, 0.12)  # thin vertical streak
bpy.ops.object.transform_apply(scale=True)

rain_mat = make_emission_mat("rain_mat", color=(0.6, 0.65, 0.9), strength=0.4)
raindrop.data.materials.append(rain_mat)

# Add particle system to emitter
rain_emitter.select_set(True)
bpy.context.view_layer.objects.active = rain_emitter
bpy.ops.object.particle_system_add()
psys = rain_emitter.particle_systems[0]
pset = psys.settings

pset.name = "rain_particles"
pset.count = 5000
pset.frame_start = 1
pset.frame_end = TOTAL_FRAMES
pset.lifetime = 240       # 10 seconds at 24fps — same as loop length → phase closes
pset.lifetime_random = 0.1
pset.emit_from = 'FACE'
pset.distribution = 'RAND'

# Physics
pset.physics_type = 'NEWTON'
pset.use_dynamic_rotation = False
pset.normal_factor = 0.0
pset.object_factor = 0.0
pset.factor_random = 0.0
pset.use_rotations = True
pset.rotation_mode = 'NOR'

# Gravity + slight wind angle (rain falls at ~80 degrees)
scene.gravity = (0.2, 0.0, -9.81)

# Size and render
pset.render_type = 'OBJECT'
pset.instance_object = raindrop
pset.particle_size = 1.0
pset.size_random = 0.2

# Fix random seed for determinism
pset.use_advanced_hair = False
# Seed for particle system
psys.seed = 12345

# ── camera ────────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(0, -6, 1.5))
cam = bpy.context.active_object
cam.name = "MainCamera"
cam.rotation_euler = (math.radians(85), 0, 0)
scene.camera = cam

cam_data = cam.data
cam_data.lens = 35.0
cam_data.clip_start = 0.1
cam_data.clip_end = 100.0

# Subtle noise-based camera shake via F-curve animation
# Use small sine wobbles baked as keyframes
cam_obj = bpy.data.objects["MainCamera"]
anim_data = cam_obj.animation_data_create()
action = bpy.data.actions.new("CameraShake")
anim_data.action = action

rng_cam = random.Random(555)
# Create F-curves for rotation X and Z (tiny shake)
fc_rx = action.fcurves.new(data_path='rotation_euler', index=0)
fc_rz = action.fcurves.new(data_path='rotation_euler', index=2)

base_rx = math.radians(85)
base_rz = 0.0
shake_amp = math.radians(0.08)  # very subtle

for f in range(1, TOTAL_FRAMES + 1):
    t = f / 24.0
    # Low-freq sinusoidal + tiny noise
    rx_val = base_rx + shake_amp * math.sin(t * 0.3 + 1.1) * 0.5
    rz_val = base_rz + shake_amp * math.sin(t * 0.2 + 0.7) * 0.3
    kp = fc_rx.keyframe_points.insert(f, rx_val)
    kp.interpolation = 'LINEAR'
    kp = fc_rz.keyframe_points.insert(f, rz_val)
    kp.interpolation = 'LINEAR'

# ── lighting: cold blue key light + dim fill ─────────────────────────────────
# Key: distant moonlight from upper-left
bpy.ops.object.light_add(type='SUN', location=(3, -3, 8))
sun = bpy.context.active_object
sun.name = "MoonLight"
sun.rotation_euler = (math.radians(50), 0, math.radians(-30))
sun.data.energy = 0.3
sun.data.color = (0.5, 0.6, 1.0)  # cold blue-white

# Fill: very dim warm-ish glow from below horizon (reflected ambient)
bpy.ops.object.light_add(type='AREA', location=(0, 0, -3))
fill = bpy.context.active_object
fill.name = "FillLight"
fill.rotation_euler = (0, math.pi, 0)
fill.data.energy = 0.05
fill.data.color = (0.4, 0.5, 0.8)
fill.data.size = 20.0

# ── output path ──────────────────────────────────────────────────────────────
scene.render.filepath = os.path.join(OUTDIR, "frame_")
scene.render.use_file_extension = True
scene.render.use_render_cache = False

print(f"[scene_rain] Scene built. Frames={TOTAL_FRAMES}, samples={SAMPLES}, outdir={OUTDIR}")
print(f"[scene_rain] Resolution: {scene.render.resolution_x}x{scene.render.resolution_y} @ {scene.render.fps}fps")

# ── trigger render from within script ────────────────────────────────────────
# This avoids the -a flag which runs before denoising is disabled
import time as _time
frame_times = []
for frame_idx in range(1, TOTAL_FRAMES + 1):
    t0 = _time.perf_counter()
    scene.frame_set(frame_idx)
    out_path = os.path.join(OUTDIR, f"frame_{frame_idx:04d}.png")
    scene.render.filepath = os.path.join(OUTDIR, f"frame_{frame_idx:04d}")
    bpy.ops.render.render(write_still=True)
    elapsed = _time.perf_counter() - t0
    frame_times.append(elapsed)
    print(f"[scene_rain] Frame {frame_idx}/{TOTAL_FRAMES} done in {elapsed:.1f}s", flush=True)

import statistics
median_t = statistics.median(frame_times)
total_t = sum(frame_times)
print(f"[scene_rain] DONE: median={median_t:.1f}s total={total_t:.0f}s frames={TOTAL_FRAMES}")
print(f"[scene_rain] TIMING_MEDIAN={median_t:.2f}")

# Write timing summary
timing_file = os.path.join(OUTDIR, "timing_summary.txt")
with open(timing_file, "w") as f:
    f.write(f"frames={TOTAL_FRAMES}\n")
    f.write(f"median_seconds={median_t:.2f}\n")
    f.write(f"total_seconds={total_t:.0f}\n")
    f.write(f"samples={SAMPLES}\n")
    f.write("per_frame_seconds=" + ",".join(f"{t:.2f}" for t in frame_times) + "\n")
