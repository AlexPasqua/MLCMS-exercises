def init_detection_zones(grid):
    grid_size = len(grid.grid)
    min_row = (grid_size // 2) - 1
    max_row = (grid_size // 2) + 1
    min_col = 14
    max_col = 16

    detection_zones = [DetectionZone(grid, min_row, max_row, min_col, max_col)]
    min_col, max_col = min_col + 25, max_col + 25
    detection_zones.append(DetectionZone(grid, min_row, max_row, min_col, max_col))
    min_col, max_col = min_col + 25, max_col + 25
    detection_zones.append(DetectionZone(grid, min_row, max_row, min_col, max_col))

    return detection_zones


def draw_detection_zones(detection_zones):
    for dz in detection_zones:
        dz.draw()


def update_densities(detection_zones):
    for dz in detection_zones:
        dz.update_density()


def get_final_metrics(detection_zones):
    avg_densities = []
    avg_speeds = []
    avg_flows = []
    for dz in detection_zones:
        dz.update_resulting_measures()
        avg_densities.append(dz.avg_density)
        avg_speeds.append(dz.avg_speed)
        avg_flows.append(dz.avg_flow)

    print(f"AVG_DENSITY: {sum(avg_densities)/len(avg_densities)}"
          f"\nAVG_SPEED: {sum(avg_speeds)/len(avg_speeds)}"
          f"\nAVG_FLOW: {sum(avg_flows)/len(avg_flows)}")


class DetectionZone:
    """
        Object to handle measurements in particular zones of the grid
        The measurements are the average density and speed over the whole simulations, making it possible to also
        measure the flow (as density*speed)
    """

    def __init__(self, grid, min_row, max_row, min_col, max_col):
        self.grid = grid
        self.min_row, self.max_row, self.min_col, self.max_col = min_row, max_row, min_col, max_col
        self.area = (self.max_row - self.min_row + 1) * (self.max_col - self.min_col + 1)
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
        :return:
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
        :return:
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
        :return:
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
        :return:
        """
        print(self, "Hey i got some speed here: ", speed)
        self.speeds.append(speed)

    def update_resulting_measures(self):
        """
        to be called once the simulation is over to get the overall measurements
        :return:
        """
        self.avg_density = sum(self.densities) / len(self.densities)
        self.avg_speed = sum(self.speeds) / len(self.speeds)
        self.avg_flow = self.avg_speed * self.avg_density

    def __str__(self):
        return f"ZONE COLUMNS: {self.min_col} - {self.max_col}"

