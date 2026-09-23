#!/usr/bin/env python3
import numpy as np
from pathlib import Path

def save_pgm(path, image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")

    height, width = image.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(image.tobytes())

# ------------------------------------------------------------------
# MAP METRICS AND GRID DEFINITION
# The map is discretized into a 5 cm occupancy grid that matches the ROS
# planner resolution and keeps the working area centered around the origin.
# ------------------------------------------------------------------
resolution = 0.05  # 5 cm per pixel
width, height = 12.0, 12.0  # meters
origin_x, origin_y = -6.0, -6.0

# ------------------------------------------------------------------
# OCCUPANCY GRID INITIALIZATION
# Free cells use a value of 0 while obstacles are marked with 100 so the
# planner can treat blocked regions as non-traversable.
# ------------------------------------------------------------------
grid = np.zeros((int(height/resolution), int(width/resolution)), dtype=np.int8)

# ------------------------------------------------------------------
# WALL BOUNDING BOXES
# These rectangles are derived from the SDF layout and represent the main walls
# and partitions in the stage. Each tuple stores the minimum and maximum x/y
# limits for a rectangular obstacle region.
# ------------------------------------------------------------------
walls = [
    (-5.6, 5.4, 5.6, 5.6),    # outer_wall_top
    (-5.6, -5.6, 5.6, -5.4),  # outer_wall_bottom
    (5.4, -5.5, 5.6, 5.5),    # outer_wall_left
    (-5.6, -5.5, -5.4, 5.5),  # outer_wall_right
    (-1.5, 2.4, 1.5, 2.6),    # wall_center_top
    (-1.5, -2.6, 1.5, -2.4),  # wall_center_bottom
    (2.9, 2.5, 3.1, 5.5),     # wall_top_left_vert
    (2.9, -5.5, 3.1, -2.5),   # wall_bottom_left
    (-3.25, 2.9, -1.75, 3.1), # wall_top_right
    (-3.1, -5.5, -2.9, -2.5), # wall_bottom_right_vert
]

# ------------------------------------------------------------------
# OBSTACLE INFLATION
# A modest inflation margin keeps the route slightly more conservative and
# preserves enough clearance around doorways for the local planner.
# ------------------------------------------------------------------
inflate = int(0.25 / resolution)

for x1, y1, x2, y2 in walls:
    # Convert the world-space wall rectangle into grid indices.
    gx1 = int((min(x1, x2) - origin_x) / resolution)
    gy1 = int((min(y1, y2) - origin_y) / resolution)
    gx2 = int((max(x1, x2) - origin_x) / resolution)
    gy2 = int((max(y1, y2) - origin_y) / resolution)

    # Mark the occupied footprint, expanded by the inflation radius.
    grid[max(0, gy1-inflate):min(grid.shape[0], gy2+inflate),
         max(0, gx1-inflate):min(grid.shape[1], gx2+inflate)] = 100

# ------------------------------------------------------------------
# MAP EXPORT FOR ROS
# The occupancy grid is saved in PGM format so ROS map_server can load it
# without additional conversion steps.
# ------------------------------------------------------------------
output_dir = Path(__file__).resolve().parent.parent / "maps"
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the target directory exists before writing.

output_path = output_dir / "stage4_map1.pgm"
legacy_output_path = output_dir / "stage4_map.pgm"

save_pgm(output_path, 255 - (grid * 2.55))
save_pgm(legacy_output_path, 255 - (grid * 2.55))
print(f"Maps saved successfully to {output_dir}")