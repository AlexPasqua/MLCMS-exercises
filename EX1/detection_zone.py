class DetectionZone:
    """
    Object to handle measurements in particular zones of the grid
    The measurements are the average density and speed over the whole simulations, making it possible to also
    measure the flow (as density*speed)
    """

    def __init__(self, grid, min_row, max_row, min_col, max_col):
        self.grid = grid    # the grid to which the detection zone belongs
        self.min_row, self.max_row, self.min_col, self.max_col = min_row, max_row, min_col, max_col
        self.area = (self.max_row - self.min_row + 1) * (self.max_col - self.min_col + 1)

        # save the cells of the grid belonging to the detection zone
        self.cells = []
        for i in range(self.min_row, self.max_row + 1):
            for j in range(self.min_col, self.max_col + 1):
                self.cells.append((i, j))

        # for coloring
        self.FILLED_COLOR_BG = "yellow"
        self.FILLED_COLOR_BORDER = "yellow"

        # in-simulation measurements for this single zone
        self.densities = []
        self.speeds = []

        # final measurements for this single zone
        self.avg_density = 0
        self.avg_speed = 0
        self.avg_flow = 0

    def trim_from_empty_time_steps(self):
        """
        it might happen (for example to the leftmost detection zone) that from a certain point on there are not
        pedestrians traversing the zone anymore. This leads to a long trail of 0 density values. Getting rid of
        these maintains the simulation in the RiMEA 4 desired situation.
        """
        to_trim = 0
        for d in reversed(self.densities):
            if d != 0:
                break
            to_trim += 1
        self.densities = self.densities[:-to_trim]

    def draw(self):
        """
        color the detection zone, paying attention to not paint over pedestrians
        """
        for cell in self.cells:
            cell_to_color = self.grid.grid[cell[0]][cell[1]]
            if cell_to_color.status != "Person":
                if not cell_to_color.fill:
                    cell_to_color.switch()
                cell_to_color.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)

    def update_density(self):
        """
        calculates the density inside a detection zone in a determinate screenshot in time.
        the calculation is simply the number of Pedestrians inside the zone divided by the area
        """
        num_pedestrians = 0
        for cell in self.cells:
            if self.grid.grid[cell[0]][cell[1]].status == 'Person':
                num_pedestrians += 1
        self.densities.append(num_pedestrians/self.area)

    def update_speed(self, speed):
        """
        when a pedestrian exits from a detection zone, he will send the DetectionZone his speed in travelling it
        :param speed: pedestrian speed, calculate as space in zone / time in zone
        """
        self.speeds.append(speed)

    def update_resulting_measures(self):
        """
        to be called once the simulation is over to get the overall measurements
        """
        self.avg_density = sum(self.densities) / len(self.densities)
        self.avg_speed = sum(self.speeds) / len(self.speeds)
        self.avg_flow = self.avg_speed * self.avg_density

    def __str__(self):
        return f"ZONE COLUMNS: {self.min_col} - {self.max_col}"

