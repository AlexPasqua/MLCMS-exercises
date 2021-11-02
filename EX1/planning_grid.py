

class PlanningGrid:
    """
    An object helping with the simulation handling, keeping it simpler to manage wrt using directly the graphical grid
    """
    BLANK_CELL = 0
    PEDESTRIAN_CELL = 1
    OBSTACLE_CELL = 2
    TARGET_CELL = 3

    def __init__(self, cell_grid):
        self.targets_list = []
        self.pedestrian_list = []
        self.obstacles_list = []
        self.grid = []
        self.init_grid(cell_grid)

    def init_grid(self, cell_grid):
        """
        Construct a much simpler grid wrt the graphical purpose one.
        Planning grid is a matrix where each cell content is an integer identifying what is inside the cell
        :param cell_grid:
        :return:
        """
        for row in range(len(cell_grid.grid)):
            grid_line = []
            for column in range(len(cell_grid.grid[row])):
                cell_status = cell_grid.grid[row][column].status
                if cell_status == 'Blank':
                    grid_line.append(self.BLANK_CELL)
                if cell_status == 'Person':
                    grid_line.append(self.PEDESTRIAN_CELL)
                    self.pedestrian_list.append((row, column))  # to maintain a list of the pedestrian coordinates
                if cell_status == 'Obstacle':
                    grid_line.append(self.OBSTACLE_CELL)
                    self.obstacles_list.append((row, column))  # to maintain a list of the obstacles coordinates
                if cell_status == 'Target':
                    grid_line.append(self.TARGET_CELL)
                    self.targets_list.append((row, column))  # to maintain a list of the targets coordinates
            self.grid.append(grid_line)

    def update_pedestrian_list(self, old_coord, new_coord):
        """
        Update a particular pedestrian position due to movement
        :param old_coord:
        :param new_coord:
        :return:
        """
        self.pedestrian_list.remove(old_coord)
        self.pedestrian_list.append(new_coord)