import time
import numpy as np
import math

from grid import CellGrid, Cell


class Pedestrian:
    def __init__(self, grid_width: int, grid_height: int, grid: CellGrid, cell: Cell):
        self.x = None
        self.y = None
        self.coords = np.array([self.x, self.y])
        self.grid = grid
        self.cell = cell
        self.active = True
        self.time_last_step = 0.
        self.waiting_time = 0.
        self.cost_matrix = [[0. for i in range(grid_height)] for j in range(grid_width)]

    def update_cost_function(self, grid):
        self.grid = grid

        # find the nearest target
        min_dist = math.inf
        min_idx = 0
        for i, target in enumerate(self.grid.targets_list):
            target_pos = np.array([target.ord, target.abs])
            dist = np.linalg.norm(target_pos - self.coords)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        nearest_target = self.grid.targets_list[min_idx]
        nearest_target_pos = np.array([nearest_target.ord, nearest_target.abs])

        # update the cost for each cell
        for line in self.grid:
            for cell in line:
                cell_pos = np.array([cell.ord, cell.abs])
                self.cost_matrix[cell.ord][cell.abs] = np.linalg.norm(nearest_target_pos - cell_pos)

        # manage targets
        for target in self.grid.targets_list:
            self.cost_matrix[target.ord][target.abs] = 0.

        # manage pedestrians
        for ped in self.grid.pedestrian_list:
            self.cost_matrix[ped.ord][ped.abs] = math.inf

        # manage obstacles
        for obs in self.grid.obstacles_list:
            self.cost_matrix[obs.ord][obs.abs] = math.inf

    def move(self, grid, timestamp):
        self.grid = grid

        if self.active:
            # plan the move
            surrounding_costs = np.zeros(shape=(3, 3))
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if not (i == 0 and j == 0):
                        if 0 <= self.x + i < len(self.grid[0]) and 0 <= self.y + j < len(self.grid):
                            surrounding_costs[i + 1][j + 1] = self.cost_matrix[self.x + i][self.y + j]
                        else:
                            surrounding_costs[i + 1][j + 1] = math.inf
            min_cost_x, min_cost_y = np.unravel_index(surrounding_costs.argmin(), surrounding_costs.shape)

            # if move is diagonal, set waiting time to 1.4s, otherwise to 1.0s
            self.waiting_time = 1.4 if abs(min_cost_x) - abs(min_cost_y) == 0 else 1.0

            # set pedestrian to sleep
            self.active = False
        else:
            self.waiting_time -= self.grid.TIME_STEP
            if self.waiting_time <= 0:
                self.active = True
