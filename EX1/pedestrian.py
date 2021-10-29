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
            # TODO: plan the move (copy code from grid ('next_movement' method I think))

            # set time of the last step and waiting time
            self.time_last_step = time.time() * 1e3
            # TODO: complete
            """
            Pseudo code:
            if move is diagonal:
                self.waiting_time = 1400 ms
            else:
                self.waiting_time = 1000 ms
            """

            # set pedestrian to sleep
            self.active = False
        else:
            self.waiting_time -= time.time() * 1e3 - self.time_last_step    # decrease waiting time

