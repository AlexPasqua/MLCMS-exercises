import time

import numpy as np
import math
import pandas as pd


class Pedestrian:
    """
    Object wrapping around Cell, allows for the agent's planning
    """

    def __init__(self, grid, cell):
        self.row = cell.ord
        self.col = cell.abs
        self.delta_row = None  # for planned movement in row axis
        self.delta_col = None  # for planned movement in col axis
        self.coords = np.array([self.row, self.col])
        self.grid = grid  # graphical grid
        self.planning_grid = None  # planning grid
        self.cell = cell
        self.active = True  # active -> ready to plan next move, not active -> waiting to actuate a planned move
        self.goal_achieved = False  # arrived at Target, waiting for object removal
        self.waiting_time = 0.  # time left to wait
        self.total_time = 0.  # total time to reach target
        # local cost function, for understanding which is the next best step to take
        self.cost_matrix = [[0. for i in range(len(grid.grid))] for j in range(len(grid.grid[0]))]

    def update_cost_function(self, planning_grid, dijkstra=True):
        """
        updates the cost matrix, taking care of particular cases such as edges, Obstacles, other Pedestrians and Targets
        :param planning_grid: (PlanningGrid object) grid used to plan the move where cells that will be occupied result occupied immediately
        :param dijkstra: if True, use Dijkstra algorithm for the shortest path, otherwise use simply euclidean distance
        """
        if not self.active:
            return

        self.planning_grid = planning_grid

        if dijkstra:
            table = pd.DataFrame(columns=("cell", "dist_from_source", "prev_cell", "visited"))

            source_col, source_row, source_name = self.col, self.row, str(self.row) + ',' + str(self.col)
            for i in range(len(self.planning_grid.grid)):
                for j in range(len(self.planning_grid.grid[0])):
                    cell_name = str(i) + ',' + str(j)  # row-col of a cell expressed as string (to give the cell a name)
                    table_row = {'cell': cell_name, 'dist_from_source': math.inf, 'prev_cell': '', 'visited': False}
                    table = table.append(table_row, ignore_index=True)
                    self.cost_matrix[i][j] = math.inf

            table.loc[table['cell'] == source_name, 'dist_from_source'] = 0  # set distance from source to itself to 0
            for i in range(len(self.planning_grid.grid)):
                for j in range(len(self.planning_grid.grid[0])):
                    cell_name = str(i) + ',' + str(j)
                    # if (self.planning_grid.grid[i][j] == self.planning_grid.PEDESTRIAN_CELL and cell_name != source_name) \
                    #         or self.planning_grid.grid[i][j] == self.planning_grid.OBSTACLE_CELL:
                    if self.planning_grid.grid[i][j] == self.planning_grid.OBSTACLE_CELL:
                        table.loc[table['cell'] == cell_name, 'dist_from_source'] = math.inf
                        table.loc[table['cell'] == cell_name, 'visited'] = True

            found_target = False
            while not found_target:
                curr_name = table[table['visited'] == False].sort_values(by='dist_from_source')['cell'].values[0]
                curr_row, curr_col = curr_name.split(',')
                curr_row, curr_col = int(curr_row), int(curr_col)
                table.loc[table['cell'] == curr_name, 'visited'] = True   # set source as visited
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if 0 <= curr_row + i < len(self.grid.grid) and 0 <= curr_col + j < len(self.grid.grid[0]):
                            neigh_row, neigh_col, neigh_name = curr_row + i, curr_col + j, str(curr_row + i) + ',' + str(curr_col + j)
                            neigh_dist = 1.4 if abs(i) == abs(j) else 1.
                            if self.planning_grid.grid[neigh_row][neigh_col] == self.planning_grid.TARGET_CELL:
                                candidate_target_dist = neigh_dist + table[table['cell'] == curr_name]['dist_from_source'].values[0]
                                if found_target:  # target already found, update best target only if nearer
                                    if target_dist > candidate_target_dist:
                                        target_dist = candidate_target_dist
                                        target_row, target_col, target_name = neigh_row, neigh_col, neigh_name
                                else:  # target not found
                                    target_dist = candidate_target_dist
                                    target_row, target_col, target_name = neigh_row, neigh_col, neigh_name
                                found_target = True
                            if neigh_name not in table[table['visited']]['cell']:   # if neighbor not visited
                                dist_from_source = neigh_dist + table[table['cell'] == curr_name]['dist_from_source'].values[0]
                                if dist_from_source < table[table['cell'] == neigh_name]['dist_from_source'].values[0]:
                                    table.loc[table['cell'] == neigh_name, 'dist_from_source'] = dist_from_source
                                    table.loc[table['cell'] == neigh_name, 'prev_cell'] = curr_name

            # create path on the cost matrix to be used in "move"
            curr_cost = 0
            curr_name = target_name
            self.cost_matrix[target_row][target_col] = curr_cost
            while curr_name != source_name:
                curr_cost += 1
                curr_name = table[table['cell'] == curr_name]['prev_cell'].values[0]
                try:
                    curr_row, curr_col = curr_name.split(',')
                except ValueError:
                    # pedestrian is stuck, cannot see the end at the moment, stand still
                    break
                curr_row, curr_col = int(curr_row), int(curr_col)
                self.cost_matrix[curr_row][curr_col] = curr_cost
        else:
            # find the nearest target by Euclidean Distance
            min_dist = math.inf
            min_idx = 0
            for i, target in enumerate(self.planning_grid.targets_list):  # target is a tuple -> (row, column)
                target_pos = np.array([target[0], target[1]])
                dist = np.linalg.norm(target_pos - self.coords)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            nearest_target = self.planning_grid.targets_list[min_idx]
            nearest_target_pos = np.array([nearest_target[0], nearest_target[1]])

            # update the cost for each cell
            for row in range(len(self.planning_grid.grid)):
                for column in range(len(self.planning_grid.grid[0])):
                    cell_pos = np.array([row, column])
                    self.cost_matrix[row][column] = np.linalg.norm(nearest_target_pos - cell_pos)

            # manage targets -> cost = 0
            for target in self.planning_grid.targets_list:
                self.cost_matrix[target[0]][target[1]] = 0.

            # manage obstacles -> cost = infinite
            for obs in self.planning_grid.obstacles_list:
                self.cost_matrix[obs[0]][obs[1]] = math.inf

        # manage pedestrians -> cost = infinite
        for ped in self.planning_grid.pedestrian_list:
            self.cost_matrix[ped[0]][ped[1]] = math.inf

    def plan_move(self):
        """
        search for the lowest costing move in the cells surrounding the Pedestrian
        :return:
        """
        surrounding_costs = np.zeros(shape=(3, 3))
        for delta_row in range(-1, 2):
            for delta_col in range(-1, 2):
                if not (delta_row == 0 and delta_col == 0):  # Pedestrian is in the center -> (0,0)
                    # check for borders and construct surrounding cost matrix
                    if 0 <= self.col + delta_col < len(self.grid.grid[0]) and 0 <= self.row + delta_row < len(self.grid.grid[:][0]):
                        surrounding_costs[delta_row + 1][delta_col + 1] = self.cost_matrix[self.row + delta_row][self.col + delta_col]
                    else:
                        surrounding_costs[delta_row + 1][delta_col + 1] = math.inf  # if out of the matrix
                else:
                    surrounding_costs[delta_row + 1][delta_col + 1] = math.inf  # cost of not moving is high
        min_cost_row, min_cost_col = np.unravel_index(surrounding_costs.argmin(), surrounding_costs.shape)

        # if all the surrounding cells have an infinite cost
        # (e.g. pedestrian surrounded by obstacles/other pedestrians) then don't move
        if surrounding_costs[0, 0] == math.inf and np.all(surrounding_costs == surrounding_costs[0, 0]):
            return 0, 0

        return min_cost_row - 1, min_cost_col - 1

    def actuate_move(self, grid, delta_row, delta_col, planning=True):
        """
        move the Pedestrian either in the planning grid or in the graphical grid
        :param grid:
        :param delta_col:
        :param delta_row:
        :param planning:
        :return:
        """
        if planning:
            self.planning_grid.grid[self.row][self.col] = self.planning_grid.BLANK_CELL  # update current cell
            # if the next cell is not a Target then move there, otherwise remove the Pedestrian
            if self.planning_grid.grid[self.row + delta_row][self.col + delta_col] != self.planning_grid.TARGET_CELL:
                self.planning_grid.grid[self.row + delta_row][self.col + delta_col] = self.planning_grid.PEDESTRIAN_CELL
                self.planning_grid.update_pedestrian_list((self.row, self.col),
                                                          (self.row + self.delta_row, self.col + self.delta_col))
            else:
                self.planning_grid.pedestrian_list.remove((self.row, self.col))

        else:
            candidate_cell = grid[self.row + delta_row][self.col + delta_col]
            # make the current cell white
            grid[self.row][self.col].switch()
            grid[self.row][self.col].draw(self.grid.FILLED_COLOR_BG, self.grid.FILLED_COLOR_BORDER)

            # color the cell we're going to (if it is not a target)
            if not candidate_cell.status == "Target":
                candidate_cell.switch()
                candidate_cell.draw(self.grid.FILLED_COLOR_BG, self.grid.FILLED_COLOR_BORDER)
            else:
                # reached the objective
                self.goal_achieved = True

    def move(self):
        """
        Function handling the planning, waiting and moving of a Pedestrian
        :return: updated planning_grid, status to understand if objective is reached or not
        """
        if self.active:
            # plan the move
            self.delta_row, self.delta_col = self.plan_move()

            # move in the planning grid
            self.actuate_move(self.planning_grid.grid, self.delta_row, self.delta_col, planning=True)

            # if move is diagonal, set waiting time to 1.4s, otherwise to 1.0s
            self.waiting_time = 1.4 if abs(self.delta_col) - abs(self.delta_row) == 0 else 1.0

            # set pedestrian to sleep
            self.active = False

            # if the pedestrian decided not to move, it stays active
            if self.delta_col == self.delta_row == 0:
                self.total_time += self.grid.TIME_STEP
                self.active = True
        else:
            # decrease the waiting time
            self.waiting_time -= self.grid.TIME_STEP
            self.total_time += self.grid.TIME_STEP
            if round(self.waiting_time, 2) <= 0:
                # if waiting time is over, move, set to active and update position
                self.actuate_move(self.grid.grid, self.delta_row, self.delta_col, planning=False)
                self.col, self.row = self.col + self.delta_col, self.row + self.delta_row
                self.active = True

        return self.planning_grid, self.goal_achieved
