import time
import numpy as np
import math

from grid import CellGrid, Cell


class Pedestrian:
    def __init__(self, grid_width: int, grid_height: int, grid: CellGrid, cell: Cell):
        self.x = None
        self.y = None
        self.next_x = None
        self.next_y = None
        self.coords = np.array([self.x, self.y])
        self.grid = grid
        self.planning_grid = None
        self.cell = cell
        self.active = True
        self.time_last_step = 0.
        self.waiting_time = 0.
        self.cost_matrix = [[0. for i in range(grid_height)] for j in range(grid_width)]

    def update_cost_function(self, planning_grid):
        self.planning_grid = planning_grid

        # find the nearest target
        min_dist = math.inf
        min_idx = 0
        for i, target in enumerate(self.planning_grid.targets_list):
            target_pos = np.array([target.ord, target.abs])
            dist = np.linalg.norm(target_pos - self.coords)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        nearest_target = self.planning_grid.targets_list[min_idx]
        nearest_target_pos = np.array([nearest_target.ord, nearest_target.abs])

        # update the cost for each cell
        for line in self.planning_grid:
            for cell in line:
                cell_pos = np.array([cell.ord, cell.abs])
                self.cost_matrix[cell.ord][cell.abs] = np.linalg.norm(nearest_target_pos - cell_pos)

        # manage targets
        for target in self.planning_grid.targets_list:
            self.cost_matrix[target.ord][target.abs] = 0.

        # manage pedestrians
        for ped in self.planning_grid.pedestrian_list:
            self.cost_matrix[ped.ord][ped.abs] = math.inf

        # manage obstacles
        for obs in self.planning_grid.obstacles_list:
            self.cost_matrix[obs.ord][obs.abs] = math.inf

    def plan_move(self):
        surrounding_costs = np.zeros(shape=(3, 3))
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not (i == 0 and j == 0):
                    if 0 <= self.x + i < len(self.grid[0]) and 0 <= self.y + j < len(self.grid[:][0]):
                        surrounding_costs[i + 1][j + 1] = self.cost_matrix[self.x + i][self.y + j]
                    else:
                        surrounding_costs[i + 1][j + 1] = math.inf
                else:
                    surrounding_costs[i + 1][j + 1] = math.inf  # cost of not moving is high
        min_cost_x, min_cost_y = np.unravel_index(surrounding_costs.argmin(), surrounding_costs.shape)

        # if all the surrounding cells have an infinite cost (e.g. pedestrian surrounded by obstacles/other pedestrians), don't move
        if surrounding_costs[0, 0] == math.inf and np.all(surrounding_costs == surrounding_costs[0, 0]):
            return 0, 0

        return min_cost_x - 1, min_cost_y - 1

    def actuate_move(self, grid: CellGrid, delta_x, delta_y):
        candidate_cell = grid[self.x + delta_x][self.y + delta_y]

        # make the current cell white
        grid[self.x][self.y].switch()
        grid[self.x][self.y].draw(self.grid.FILLED_COLOR_BG, self.grid.FILLED_COLOR_BORDER)

        # color the cell we're going to
        candidate_cell.switch()
        candidate_cell.draw(self.grid.FILLED_COLOR_BG, self.grid.FILLED_COLOR_BORDER)

    def move(self):
        if self.active:
            # plan the move
            self.next_x, self.next_y = self.plan_move()

            # move in the planning grid
            self.actuate_move(self.planning_grid, self.next_x, self.next_y)

            # if move is diagonal, set waiting time to 1.4s, otherwise to 1.0s
            self.waiting_time = 1.4 if abs(self.next_x) - abs(self.next_y) == 0 else 1.0

            # set pedestrian to sleep
            self.active = False

            # if the pedestrian decided not to move, it stays active
            if self.next_x == self.next_y == 0:
                self.active = True
        else:
            self.waiting_time -= self.grid.TIME_STEP
            if self.waiting_time <= 0:
                self.actuate_move(self.grid, self.next_x, self.next_y)
                self.x, self.y = self.next_x, self.next_y
                self.active = True

        return self.planning_grid
