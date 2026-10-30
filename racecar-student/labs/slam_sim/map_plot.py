import os
import math

import matplotlib
matplotlib.use("Agg") #neededc for some reason
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def save_map_plot(map_msg, pose, goal_xy, trail, path):
    w, h, res = map_msg.info.width, map_msg.info.height, map_msg.info.resolution
    ox, oy = map_msg.info.origin.position.x, map_msg.info.origin.position.y
    data = np.asarray(map_msg.data, dtype=np.int16).reshape(h, w) #yx

    img = np.zeros((h, w), dtype=np.uint8)
    img[(data >= 0) & (data < 50)] = 1
    img[data >= 50] = 2
    cmap = ListedColormap(["#b0b0b0", "#ffffff", "#000000"]) #0 unknown g, 1 free w, 2 occ b

    known = np.argwhere(img > 0) #zoom so not full mapo
    margin = int(1.0 / res)
    if len(known):
        y0, x0 = known.min(axis=0) - margin
        y1, x1 = known.max(axis=0) + margin
    else:
        y0, x0, y1, x1 = 0, 0, h, w
    xmin, xmax = ox + max(x0, 0) * res, ox + min(x1 + 1, w) * res
    ymin, ymax = oy + max(y0, 0) * res, oy + min(y1 + 1, h) * res

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=cmap, vmin=0, vmax=2, origin="lower",
              extent=(ox, ox + w * res, oy, oy + h * res), interpolation="nearest")

    if len(trail) > 1: #exist
        tx, ty = zip(*trail)
        ax.plot(tx, ty, "-", color="tab:blue", linewidth=1.5, label="estimated path")
    ax.plot(*goal_xy, marker="*", color="tab:green", markersize=16, linestyle="none", label="goal")
    ax.arrow(pose.x, pose.y, 0.3 * math.cos(pose.theta), 0.3 * math.sin(pose.theta),
             head_width=0.12, color="tab:red", zorder=5)
    ax.plot(pose.x, pose.y, "o", color="tab:red", markersize=7, label="car", zorder=5)

    ax.set_xlim(xmin, xmax) #scaje
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, start frame)")
    ax.set_ylabel("y (m, start frame)")
    ax.set_title(f"Occupancy grid  t={map_msg.header.stamp.nanoseconds * 1e-9:.1f}s  pose=({pose.x:.2f}, {pose.y:.2f}, {pose.theta:.2f} rad)") #info needed
    ax.legend(loc="upper right", fontsize=8)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig) #dont store forever