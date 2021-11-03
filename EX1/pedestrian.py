import numpy as np
import math
import pandas as pd


class Pedestrian:
    """
    Object wrapping around Cell, allows for the agent's planning
    """

    def __init__(self, grid, cell):
        self.x = cell.abs
        self.y = cell.ord
        self.delta_x = None  # for planned movement in x axis
        self.delta_y = None  # for planned movement in y axis
        self.coords = np.array([self.y, self.x])
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
        self.planning_grid = planning_grid

        if dijkstra:
            # create lists of visited and unvisited cells initialize all costs to infinity
            visited, unvisited = [], []
            source_x, source_y, source_name = self.x, self.y, str(self.x) + ',' + str(self.y)
            for i in range(len(self.planning_grid.grid[:][0])):
                for j in range(len(self.planning_grid.grid[0][:])):
                    cell_coords_as_str = str(i) + ',' + str(j)  # coords of a cell expressed as string (to give the cell a name)
                    unvisited.append(cell_coords_as_str)
                    self.cost_matrix[i][j] = math.inf
                    if self.planning_grid.grid[i][j] == self.planning_grid.TARGET_CELL:
                        target_x, target_y = i, j
                        target_name = cell_coords_as_str

            # create table: cell - distance from source - prev cell
            table = pd.DataFrame(columns=("cell", "dist_from_source", "prev_cell"))
            for cell in unvisited:
                table = table.append({'cell': cell, 'dist_from_source': math.inf, 'prev_cell': '-'}, ignore_index=True)

            # for current cell, examine unvisited neighbors - if distance is shorter, update the table
            curr_x, curr_y, curr_name = source_x, source_y, source_name
            # table[table['cell'] == source_name]['dist_from_source'] = 0 # set distance from source to itself to 0
            visited.append(curr_name)
            unvisited.remove(curr_name)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= curr_x + i < len(self.grid.grid[:][0]) and 0 <= curr_y + j < len(self.grid.grid[0][:]):
                        neigh_x, neigh_y, neigh_name = i, j, str(curr_x + i) + ',' + str(curr_y + j)
                        neigh_dist = 1.4 if abs(i) == abs(j) else 1.
                        dist_from_source = neigh_dist + table[table['cell'] == curr_name]['dist_from_source'].values[0]
                        prev_cell = table[table['cell'] == neigh_name]['prev_cell'].values[0]
                        if dist_from_source < table[table['cell'] == neigh_name]['dist_from_source'].values[0]:
                            prev_cell = curr_name
                        table.loc[table['cell'] == neigh_name, 'dist_from_source'] = dist_from_source
                        table.loc[table['cell'] == neigh_name, 'prev_cell'] = prev_cell
            print(table.head())
            exit()
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

            # manage pedestrians -> cost = infinite
            for ped in self.planning_grid.pedestrian_list:
                self.cost_matrix[ped[0]][ped[1]] = math.inf

            # manage obstacles -> cost = infinite
            for obs in self.planning_grid.obstacles_list:
                self.cost_matrix[obs[0]][obs[1]] = math.inf

    def plan_move(self):
        """
        search for the lowest costing move in the cells surrounding the Pedestrian
        :return:
        """
        surrounding_costs = np.zeros(shape=(3, 3))
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not (i == 0 and j == 0):  # Pedestrian is in the center -> (0,0)
                    # check for borders and construct surrounding cost matrix
                    if 0 <= self.x + i < len(self.grid.grid[0]) and 0 <= self.y + j < len( self.grid.grid[:][0]):
                        surrounding_costs[i + 1][j + 1] = self.cost_matrix[self.y + j][self.x + i]
                    else:
                        surrounding_costs[i + 1][j + 1] = math.inf  # if out of the matrix
                else:
                    surrounding_costs[i + 1][j + 1] = math.inf  # cost of not moving is high
        min_cost_x, min_cost_y = np.unravel_index(surrounding_costs.argmin(), surrounding_costs.shape)

        # if all the surrounding cells have an infinite cost
        # (e.g. pedestrian surrounded by obstacles/other pedestrians) then don't move
        if surrounding_costs[0, 0] == math.inf and np.all(surrounding_costs == surrounding_costs[0, 0]):
            return 0, 0
        return min_cost_x - 1, min_cost_y - 1

    def actuate_move(self, grid, delta_x, delta_y, planning=True):
        """
        move the Pedestrian either in the planning grid or in the graphical grid
        :param grid:
        :param delta_x:
        :param delta_y:
        :param planning:
        :return:
        """
        if planning:
            self.planning_grid.grid[self.y][self.x] = self.planning_grid.BLANK_CELL  # update current cell
            # if the next cell is not a Target then move there, otherwise remove the Pedestrian
            if self.planning_grid.grid[self.y + delta_y][self.x + delta_x] != self.planning_grid.TARGET_CELL:
                self.planning_grid.grid[self.y + delta_y][self.x + delta_x] = self.planning_grid.PEDESTRIAN_CELL
                self.planning_grid.update_pedestrian_list((self.y, self.x),
                                                          (self.y + self.delta_y, self.x + self.delta_x))
            else:
                self.planning_grid.pedestrian_list.remove((self.y, self.x))

        else:
            candidate_cell = grid[self.y + delta_y][self.x + delta_x]
            # make the current cell white
            grid[self.y][self.x].switch()
            grid[self.y][self.x].draw(self.grid.FILLED_COLOR_BG, self.grid.FILLED_COLOR_BORDER)

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
            self.delta_x, self.delta_y = self.plan_move()

            # move in the planning grid
            self.actuate_move(self.planning_grid.grid, self.delta_x, self.delta_y, planning=True)

            # if move is diagonal, set waiting time to 1.4s, otherwise to 1.0s
            self.waiting_time = 1.4 if abs(self.delta_x) - abs(self.delta_y) == 0 else 1.0

            # set pedestrian to sleep
            self.active = False

            # if the pedestrian decided not to move, it stays active
            if self.delta_x == self.delta_y == 0:
                self.active = True
        else:
            # decrease the waiting time
            self.waiting_time -= self.grid.TIME_STEP
            self.total_time += self.grid.TIME_STEP
            if round(self.waiting_time, 2) <= 0:
                # if waiting time is over, move, set to active and update position
                self.actuate_move(self.grid.grid, self.delta_x, self.delta_y, planning=False)
                self.x, self.y = self.x + self.delta_x, self.y + self.delta_y
                self.active = True

        return self.planning_grid, self.goal_achieved
