"""Build a configurable static indoor scene without modifying pinned upstream assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]


def numbers(values):
    return " ".join(f"{v:.8g}" for v in values)


def build(config_path, output):
    c = json.loads(config_path.read_text())
    r, t = c["room"], c["threshold"]
    if not (r["x_min"] < 0 < r["x_max"] and r["width"] > 0 and r["wall_height"] > 0):
        raise ValueError("room must contain the origin and have positive dimensions")
    friction = np.asarray(c["floor_friction"], dtype=float)
    if friction.shape != (3,) or not np.isfinite(friction).all() or (friction < 0).any():
        raise ValueError("floor_friction needs three finite nonnegative values")
    source = ROOT / "upstream/src/mjlab_microduck/robot/microduck"
    scene = ET.parse(source / "scene_walk.xml").getroot()
    # Inline a copy, resolving asset paths without changing the official checkout.
    include = scene.find("include")
    robot = ET.parse(source / include.attrib["file"]).getroot()
    compiler = robot.find("compiler")
    compiler.set("meshdir", str(source / "assets"))
    scene.remove(include)
    for index, child in enumerate(list(robot)):
        scene.insert(index, child)
    scene.set("model", "microduck_indoor_lab")
    world = next(w for w in scene.findall("worldbody") if w.find("geom[@name='floor']") is not None)
    floor = world.find("geom[@name='floor']")
    floor.set("friction", numbers(friction))
    floor.set("priority", "1")
    floor.set("condim", "3")
    texture = scene.find("asset/texture[@name='groundplane']")
    texture.set("rgb1", "0.56 0.45 0.33")
    texture.set("rgb2", "0.64 0.53 0.40")

    def box(name, center, size, rgba):
        center, size = np.asarray(center, float), np.asarray(size, float)
        if center.shape != (2,) or size.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError(f"invalid position or dimensions for {name}")
        ET.SubElement(world, "geom", name=name, type="box",
                      pos=numbers([*center, size[2] / 2]), size=numbers(size / 2),
                      rgba=rgba, friction=numbers(friction), priority="1", condim="3")

    box("terrain_threshold", t["center_xy"], [t["length_x"], t["width_y"], t["height"]], "0.92 0.57 0.13 1")
    colors = ["0.18 0.52 0.64 1", "0.66 0.35 0.22 1", "0.37 0.52 0.32 1"]
    for b, color in zip(c["boxes"], colors, strict=True):
        box(b["name"], b["center_xy"], b["size_xyz"], color)
    mid = (r["x_min"] + r["x_max"]) / 2
    length = r["x_max"] - r["x_min"]
    for name, xy, size in [
        ("wall_back", [r["x_min"] - 0.025, 0], [0.05, r["width"] + 0.1, r["wall_height"]]),
        ("wall_front", [r["x_max"] + 0.025, 0], [0.05, r["width"] + 0.1, r["wall_height"]]),
        ("wall_left", [mid, r["width"] / 2 + 0.025], [length, 0.05, r["wall_height"]]),
        ("wall_right", [mid, -r["width"] / 2 - 0.025], [length, 0.05, r["wall_height"]]),
    ]:
        box(name, xy, size, "0.75 0.80 0.84 1")
    for name, xy, color in [("start", [0, 0], "0.1 0.8 0.5 0.45"), ("goal", c["goal_xy"], "0.15 0.5 1 0.6")]:
        ET.SubElement(world, "site", name=name, type="cylinder", pos=numbers([*xy, 0.001]),
                      size="0.13 0.001", rgba=color)
    # Static boxes add no joints: the original STAND keyframe and 14-action contract remain valid.
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(scene)
    ET.ElementTree(scene).write(output, encoding="unicode")
    return c


def preview(path, destination, config):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [1.3, 0, 0]
    camera.distance, camera.azimuth, camera.elevation = 5.4, 125, -63
    model.vis.global_.offwidth, model.vis.global_.offheight = 1600, 1100
    destination.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio
    with mujoco.Renderer(model, height=1100, width=1600) as renderer:
        renderer.update_scene(data, camera=camera)
        iio.imwrite(destination / "indoor_lab_preview.png", renderer.render())
    # A dimensioned plan supplements the physical render.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    r, t = config["room"], config["threshold"]
    ax.add_patch(Rectangle((r["x_min"], -r["width"]/2), r["x_max"]-r["x_min"], r["width"], facecolor="#f3eadc", edgecolor="#456", lw=3))
    x, y = t["center_xy"]
    ax.add_patch(Rectangle((x-t["length_x"]/2, y-t["width_y"]/2), t["length_x"], t["width_y"], color="#eaa32c"))
    ax.annotate("Threshold: 1 cm high\n10 cm deep / 120 cm wide", (x, -0.6), (0.05, -1.22), arrowprops={"arrowstyle":"->"}, fontsize=11)
    for b, color in zip(config["boxes"], ["#2e869e", "#aa5938", "#5e8552"], strict=True):
        x,y = b["center_xy"]; dx,dy,dz = b["size_xyz"]
        ax.add_patch(Rectangle((x-dx/2,y-dy/2), dx,dy,facecolor=color))
        ax.text(x,y, b['name'], ha="center",va="center",color="white",fontsize=10)
        ax.text(x,y+dy/2+.08,f"{dx*100:g} x {dy*100:g} cm / h={dz*100:g} cm",ha="center",va="bottom",fontsize=9)
    for xy, color, label in [([0,0], "#159d70", "START"), (config["goal_xy"], "#267ade", "GOAL")]:
        ax.add_patch(Circle(xy,0.13,color=color,alpha=.7))
        ax.text(xy[0],xy[1]-.24,label,ha="center",fontsize=10)
    ax.annotate("+X / forward", (0.5,0.05),(0,0.05), arrowprops={"arrowstyle":"->"}, va="bottom")
    ax.set(xlim=(-.85,3.65),ylim=(-1.7,1.7),xlabel="X (metres)",ylabel="Y (metres)",aspect="equal",title="MicroDuck indoor lab | 4 m x 3 m\nDesigned example: dimensions and friction are not measured")
    ax.grid(alpha=.2)
    fig.savefig(destination / "indoor_lab_plan.png",dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scenes/indoor_lab.json")
    parser.add_argument("--output", type=Path, default=ROOT / "scenes/indoor_lab.xml")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    config = build(args.config, args.output)
    # Compile immediately: missing meshes and invalid keyframes fail before playback.
    model = mujoco.MjModel.from_xml_path(str(args.output))
    print(f"Built {args.output}: {model.nu} actuators, {model.ngeom} geoms")
    if args.preview:
        preview(args.output, ROOT / "outputs/indoor_lab", config)
