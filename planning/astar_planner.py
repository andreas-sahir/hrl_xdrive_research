#!/usr/bin/env python3
"""
auto_explorer.py
A* path planner for the Stage 4 maze. Loads occupancy grid and plans paths.
"""

import numpy as np
import heapq
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Deque
from collections import deque

def load_pgm(path: str) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as f:
        header = f.readline().strip()
        if header != b"P5":
            raise ValueError("Unsupported PGM format: expected P5 binary PGM")
        def read_noncomment():
            while True:
                line = f.readline()
                if not line: raise ValueError("Unexpected EOF while reading PGM header")
                line = line.strip()
                if not line or line.startswith(b"#"): continue
                return line
        dims = read_noncomment().split()
        if len(dims) < 2: dims += read_noncomment().split()
        width, height = int(dims[0]), int(dims[1])
        maxval = int(read_noncomment())
        data = np.frombuffer(f.read(width * height), dtype=np.uint8)
        return data.reshape((height, width))

@dataclass
class Node:
    x: int
    y: int
    g: float = 0.0
    h: float = 0.0
    parent: Optional['Node'] = None

    @property
    def f(self): return self.g + self.h
    def __lt__(self, other): return self.f < other.f
    def __eq__(self, other): return self.x == other.x and self.y == other.y
    def __hash__(self): return hash((self.x, self.y))

class AStarPlanner:
    def __init__(self, grid_path: str, resolution: float = 0.05, 
                 origin_x: float = -6.0, origin_y: float = -6.0):
        grid_path = Path(grid_path)
        if grid_path.suffix.lower() == ".npy":
            self.grid = np.load(grid_path)
        else:
            self.grid = np.where(load_pgm(grid_path) >= 128, 0, 100).astype(np.int8)
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.height, self.width = self.grid.shape

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return int((x - self.origin_x) / self.resolution), int((y - self.origin_y) / self.resolution)

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return gx * self.resolution + self.origin_x + self.resolution/2, gy * self.resolution + self.origin_y + self.resolution/2

    def is_valid(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height and self.grid[gy, gx] == 0

    def heuristic(self, gx: int, gy: int, goal_gx: int, goal_gy: int) -> float:
        return math.sqrt((gx - goal_gx)**2 + (gy - goal_gy)**2) * self.resolution

    def get_neighbors(self, node: Node) -> List[Node]:
        neighbors = []
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = node.x + dx, node.y + dy
            if self.is_valid(nx, ny):
                cost = (math.sqrt(2) if dx != 0 and dy != 0 else 1.0) * self.resolution
                neighbors.append(Node(nx, ny, g=node.g + cost))
        return neighbors

    def _run_astar_search(self, start_world: Tuple[float, float], goal_world: Tuple[float, float]) -> List[Tuple[int, int]]:
        start_gx, start_gy = self.world_to_grid(*start_world)
        goal_gx, goal_gy = self.world_to_grid(*goal_world)
        
        if not self.is_valid(start_gx, start_gy): start_gx, start_gy = self._find_nearest_free(start_gx, start_gy)
        if not self.is_valid(goal_gx, goal_gy): goal_gx, goal_gy = self._find_nearest_free(goal_gx, goal_gy)

        open_set = [Node(start_gx, start_gy, g=0, h=self.heuristic(start_gx, start_gy, goal_gx, goal_gy))]
        came_from = {}
        g_score = {(start_gx, start_gy): 0.0}
        
        while open_set:
            current = heapq.heappop(open_set)
            if current.x == goal_gx and current.y == goal_gy:
                return self._reconstruct_grid_path(came_from, (current.x, current.y), (start_gx, start_gy))
            for neighbor in self.get_neighbors(current):
                tentative_g = neighbor.g
                if (neighbor.x, neighbor.y) not in g_score or tentative_g < g_score[(neighbor.x, neighbor.y)]:
                    came_from[(neighbor.x, neighbor.y)] = (current.x, current.y)
                    g_score[(neighbor.x, neighbor.y)] = tentative_g
                    neighbor.h = self.heuristic(neighbor.x, neighbor.y, goal_gx, goal_gy)
                    heapq.heappush(open_set, neighbor)
        return []

    def _reconstruct_grid_path(self, came_from: dict, goal: Tuple[int, int], start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [goal]
        curr = goal
        while curr != start:
            curr = came_from[curr]
            path.append(curr)
        path.reverse()
        return path

    def plan(self, start_world, goal_world, max_waypoint_spacing=1.5):
        dense_grid_path = self._run_astar_search(start_world, goal_world) 
        if not dense_grid_path: return []
        
        dense_world_path = [self.grid_to_world(cx, cy) for cx, cy in dense_grid_path]
        sparse_waypoints = [dense_world_path[0]]
        last_wp = dense_world_path[0]
        
        for pt in dense_world_path[1:]:
            if math.hypot(pt[0] - last_wp[0], pt[1] - last_wp[1]) >= max_waypoint_spacing:
                sparse_waypoints.append(pt)
                last_wp = pt
        if dense_world_path[-1] not in sparse_waypoints:
            sparse_waypoints.append(dense_world_path[-1])
        return sparse_waypoints

    def _find_nearest_free(self, gx: int, gy: int, max_radius: int = 50) -> Tuple[int, int]:
        queue = deque([(gx, gy, 0)])
        visited = {(gx, gy)}
        while queue:
            cx, cy, dist = queue.popleft()
            if self.is_valid(cx, cy): return cx, cy
            if dist < max_radius:
                for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                    nx, ny = cx+dx, cy+dy
                    if (nx, ny) not in visited and 0 <= nx < self.width and 0 <= ny < self.height:
                        visited.add((nx, ny)); queue.append((nx, ny, dist+1))
        return gx, gy

if __name__ == "__main__":
    # Test block call to verify
    planner = AStarPlanner("path/to/your/stage4_map.pgm")
    path = planner.plan((0,0), (3,3))
    print(f"Plan successful: {len(path)} waypoints")