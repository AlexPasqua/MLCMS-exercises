from tkinter import *
import time
import random
from planning_grid import *
from pedestrian import *
from cell import *
from detection_zone import *


class CellGrid(Canvas):
    """
    Object representing the grid that acts as environment for the cellular automaton.
    It is also used to handle the overall execution, maintaining the graphical grid and switching between different modalities
    """

    def __init__(self, master, row_number: int, cell_size: int, is_rimea_4=False, *args, **kwargs):
        self.canvas = Canvas.__init__(self, master, width=cell_size * row_number, height=cell_size * row_number,
                                      *args, **kwargs)
        self.master = master

        # default coloring of the cells
        self.FILLED_COLOR_BG = "green"
        self.FILLED_COLOR_BORDER = "green"

        self.selected_pedestrian = None  # needed for free walk mode

        # attributes for simulation purposes
        self.targets_cell_list = []         # list containing the Cell objects occupied by targets
        self.obstacles_cell_list = []       # list containing the Cell objects occupied by obstacles
        self.pedestrian_cell_list = []      # lower level structure, containing only the Cell objects
        self.pedestrian_list = None         # higher level structure, containing Pedestrian objects
        self.pedestrian_speeds = []         # list of pedestrian final speeds, needed for statistical aspect in RiMEA Test 7
        self.dijkstra_enabled = IntVar()    # variable associated to checkbox for enabling/disabling Dijkstra
        self.TIME_STEP = 0.1                # simulation time step (for time discretization)

        # buttons
        self.person_button = None
        self.obstacle_button = None
        self.target_button = None
        self.run_simulation_button = None
        self.free_walk_button = None
        self.dijkstra = None
        self.buttons = []
        self.init_buttons()     # bind buttons attribute to real physical buttons

        # graphical grid initialization
        self.cellSize = cell_size
        self.grid = []
        column_number = row_number
        for row in range(row_number):
            line = [Cell(self, x=column, y=row, size=cell_size) for column in range(column_number)]  # a row of the grid
            self.grid.append(line)

        # attributes for RiMEA 4
        self.is_rimea_4 = is_rimea_4    # boolean to avoid some computation if not needed for RiMEA 4
        self.detection_zones = self.init_detection_zones()   # detection zones for RiMEA 4

        # memorize the cells that have been modified to avoid many switching of state during mouse motion.
        self.switched = []

        # bind click action
        self.bind("<Button-1>", self.handle_mouse_click)
        # bind moving while clicking
        self.bind("<B1-Motion>", self.handle_mouse_motion)
        # bind release button action - clear the memory of modified cells.
        self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())
        # bind escape button action - exit application
        self.bind("<Escape>", sys.exit)

        # draw empty grid
        self.draw_empty_grid()

    def draw_empty_grid(self):
        """
        Draws the empty grid to start up the environment
        """
        for row in self.grid:
            for cell in row:
                cell.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)

    def _event_coords(self, event):
        """
        Captures the correct locations of keyboard/mouse triggered events
        :param event: the event that happened
        :return: the row and columns where the event occurred
        """
        column = int(event.x / self.cellSize)
        row = int(event.y / self.cellSize)
        return row, column

    def handle_mouse_click(self, event):
        """
        Handle left mouse click, coloring a specific cell
        """
        row, column = self._event_coords(event)
        cell = self.grid[row][column]
        cell.switch()
        cell.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)
        # add the cell to the list of cell switched during the click
        self.switched.append(cell)

    def handle_mouse_motion(self, event):
        """
        Handle the multiple coloring feature, happening when clicking and maintaining left mouse button.
        """
        row, column = self._event_coords(event)
        cell = self.grid[row][column]
        if cell not in self.switched:
            cell.switch()
            cell.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)
            self.switched.append(cell)

    def init_buttons(self):
        """
        Initialize all buttons, binding them to a certain function to trigger when an event arrives
        """
        self.person_button = Button(self.canvas, text="Person", command=self.draw_person)
        self.person_button.pack()
        self.obstacle_button = Button(self.canvas, text="Obstacle", command=self.draw_obstacle)
        self.obstacle_button.pack()
        self.target_button = Button(self.canvas, text="Target", command=self.draw_target)
        self.target_button.pack()
        self.run_simulation_button = Button(self.canvas, text="Run Simulation", command=self.start_simulation)
        self.run_simulation_button.pack()
        self.free_walk_button = Button(self.canvas, text="Free Walk", command=self.switch_mode)
        self.free_walk_button.pack()
        self.buttons = [self.person_button, self.obstacle_button, self.target_button, self.run_simulation_button,
                        self.free_walk_button]
        # target button is pressed by default
        self.target_button.configure(relief=SUNKEN, state=DISABLED)

        # setup checkbox for dijkstra enabling
        self.dijkstra = Checkbutton(self.canvas, text='Dijkstra', variable=self.dijkstra_enabled)
        self.dijkstra.pack()

    def update_buttons_state(self, current_button_text: str):
        """
        Raise the previously pressed button, keeps pressed the last fired button
        :param current_button_text: current text written in the button
        """
        for button in self.buttons:
            if button['text'] == current_button_text:
                button.configure(relief=SUNKEN, state=DISABLED)     # sink and disable the button
            elif button['state'] == DISABLED:
                button.configure(relief=RAISED, state=ACTIVE)       # raise and enable the button

    def draw_person(self):
        """
        prepares the color configuration for drawing Pedestrian
        """
        self.update_buttons_state("Person")
        color = "red"
        self.FILLED_COLOR_BG = color
        self.FILLED_COLOR_BORDER = color

    def draw_obstacle(self):
        """
        prepares the color configuration for drawing Obstacle
        """
        self.update_buttons_state("Obstacle")
        color = "black"
        self.FILLED_COLOR_BG = color
        self.FILLED_COLOR_BORDER = color

    def draw_target(self):
        """
        prepares the color configuration for drawing Target
        """
        self.update_buttons_state("Target")
        color = "green"
        self.FILLED_COLOR_BG = color
        self.FILLED_COLOR_BORDER = color

    def switch_mode(self):
        """
        handler for switching back and forth between Editing Mode and Free Walk Mode
        """
        if self.free_walk_button['text'] == 'Free Walk':
            self.free_walk_mode()
        else:
            self.editing_mode()

    def free_walk_mode(self):
        """
        changes the key bindings for handling the free walk mode where it's possible to move a pedestrian with the
        arrows on the keyboards or with the WASD keys
        """
        print('Entering movement mode..')
        # bind click action
        self.bind("<Button-1>", self.select_person)

        # bind left arrow and 'a' action
        self.bind("<Left>", self.move_person)
        self.bind("a", self.move_person)

        # bind right arrow and 'd' action
        self.bind("<Right>", self.move_person)
        self.bind("d", self.move_person)

        # bind up arrow and 'w' action
        self.bind("<Up>", self.move_person)
        self.bind("w", self.move_person)

        # bind down arrow and 's' action
        self.bind("<Down>", self.move_person)
        self.bind("s", self.move_person)

        # unbind unnecessary keys
        self.unbind("<B1-Motion>")
        self.unbind("<ButtonRelease-1>")
        self.FILLED_COLOR_BORDER = 'red'
        self.FILLED_COLOR_BG = 'red'
        self.person_button.configure(relief=SUNKEN, state=DISABLED)
        self.obstacle_button.configure(relief=SUNKEN, state=DISABLED)
        self.target_button.configure(relief=SUNKEN, state=DISABLED)
        self.free_walk_button['text'] = "Editing mode"

    def editing_mode(self):
        """
        changes the key bindings for handling the editing mode, where it's possible to modify the layout of the grid
        """
        print('Entering movement mode..')
        # bind new buttons
        self.bind("<Button-1>", self.handle_mouse_click)
        self.bind("<B1-Motion>", self.handle_mouse_motion)
        self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())
        self.unbind(("<Left>", "<Right>", "<Up>", "<Down>", "a", "s", "d", "w"))    # unbind the arrows and WASD keys
        # set un the buttons correctly on the interface
        self.person_button.configure(relief=RAISED, state=ACTIVE)       # person
        self.obstacle_button.configure(relief=RAISED, state=ACTIVE)     # obstacle
        self.target_button.configure(relief=SUNKEN, state=DISABLED)     # target
        self.draw_target()
        self.free_walk_button['text'] = "Free Walk"  # free walk / editing mode

    def select_person(self, event):
        """
        left-clicking on a Person cell will select it, making it able to be moved
        """
        row, column = self._event_coords(event)
        candidate_cell = self.grid[row][column]
        if candidate_cell.status == 'Person':
            self.selected_pedestrian = candidate_cell   # set the selected pedestrian to be able to move it

    def move_person(self, event=None):
        """
        Handle pedestrians' movements in the FREE WALK MODE
        """
        if self.selected_pedestrian is not None:  # a pedestrian has to be selected by left-clicking it
            self.selected_pedestrian.switch()
            self.selected_pedestrian.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)

            # handle movements in different directions
            if event.keysym == 'Right' or event.keysym == 'd':
                candidate_cell = self.grid[self.selected_pedestrian.ord][self.selected_pedestrian.abs + 1]
            elif event.keysym == 'Left' or event.keysym == 'a':
                candidate_cell = self.grid[self.selected_pedestrian.ord][self.selected_pedestrian.abs - 1]
            elif event.keysym == 'Up' or event.keysym == 'w':
                candidate_cell = self.grid[self.selected_pedestrian.ord - 1][self.selected_pedestrian.abs]
            else:
                candidate_cell = self.grid[self.selected_pedestrian.ord + 1][self.selected_pedestrian.abs]

            if candidate_cell.status not in ('Obstacle', 'Person'):  # to avoid going over an Obstacle or other person
                self.selected_pedestrian = candidate_cell
            if candidate_cell.status == 'Target':  # disappear when going on Target
                self.selected_pedestrian = None
            else:
                self.selected_pedestrian.switch()
                self.selected_pedestrian.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)

    def start_simulation(self):
        """
        handler for the Simulation Mode
        """
        print('Entering simulation mode..')
        # unbind unnecessary keys
        self.unbind("<Button-1>")
        self.unbind("<B1-Motion>")
        self.unbind("<ButtonRelease-1>")
        # setting the colors for pedestrians
        self.FILLED_COLOR_BORDER = 'red'
        self.FILLED_COLOR_BG = 'red'
        # configure the buttons
        self.person_button.configure(relief=SUNKEN, state=DISABLED)
        self.obstacle_button.configure(relief=SUNKEN, state=DISABLED)
        self.target_button.configure(relief=SUNKEN, state=DISABLED)
        self.free_walk_button['text'] = "Editing mode"

        # create a PlanningGrid for more efficient management of simulation
        planning_grid = PlanningGrid(self)

        # create a Pedestrian list to access useful methods ("if" used for RiMEA Test 7)
        if self.pedestrian_list is None:
            self.pedestrian_list = [Pedestrian(self, cell, is_rimea=self.is_rimea_4) for cell in self.pedestrian_cell_list]

        # continue simulating until all pedestrian have not reached a target
        activate_dijkstra = True if self.dijkstra_enabled.get() == 1 else False
        while len(self.pedestrian_list) != 0:
            random.shuffle(self.pedestrian_list)  # to avoid giving advantage to the same Pedestrian all the time
            for pedestrian in self.pedestrian_list:
                pedestrian.update_cost_function(planning_grid, dijkstra=activate_dijkstra)  # update local cost_matrix
                planning_grid, pedestrian_has_ended = pedestrian.move()  # try to move (time constraints)

                if pedestrian_has_ended:
                    self.pedestrian_speeds.append(pedestrian.total_meters / pedestrian.total_time)
                    print("Pedestrian reached target in:", round(pedestrian.total_time, 3), 'with a speed of:',
                          round(self.pedestrian_speeds[-1], 2), "(expected speed was: ", pedestrian.speed, ")")
                    self.pedestrian_list.remove(pedestrian)

            time.sleep(self.TIME_STEP)  # discretization

            if self.is_rimea_4:
                self.draw_detection_zones()
                self.update_densities()

            self.update()  # graphical update of the grid

        self.pedestrian_list = None     # necessary in case of multiple simulations without closing the app

        if self.is_rimea_4:
            self.get_final_dectzones_metrics()

    def init_detection_zones(self):
        """
        Initialize the detection zones for the RiMEA scenario 4
        :return: the list of detection zones (DetectionZone objects)
        """
        grid_size = len(self.grid)
        min_row = (grid_size // 2) - 1
        max_row = (grid_size // 2) + 1
        min_col = 14
        max_col = 16
        detection_zones = [DetectionZone(self, min_row, max_row, min_col, max_col)]
        min_col, max_col = min_col + 25, max_col + 25
        detection_zones.append(DetectionZone(self, min_row, max_row, min_col, max_col))
        min_col, max_col = min_col + 25, max_col + 25
        detection_zones.append(DetectionZone(self, min_row, max_row, min_col, max_col))
        return detection_zones

    def draw_detection_zones(self):
        """ Draw the detection zones on teh grid """
        for dz in self.detection_zones:
            dz.draw()

    def update_densities(self):
        """ Update the pedestrian density of each detection zone by calling its apposite method """
        for dz in self.detection_zones:
            dz.update_density()     # update pedestrian density for the current detection zone

    def get_final_dectzones_metrics(self):
        """
        Get the final metrics from the detection zones: average speed, density and flow.
        These are obtained getting the average metrics during the simulation for each detection zone and doing
        the average per number of detection zones
        """
        avg_densities = []
        avg_speeds = []
        avg_flows = []
        for dz in self.detection_zones:
            dz.update_resulting_measures()
            avg_densities.append(dz.avg_density)
            avg_speeds.append(dz.avg_speed)
            avg_flows.append(dz.avg_flow)

        # print results
        print(f"AVG_DENSITY: {sum(avg_densities) / len(avg_densities)}"
              f"\nAVG_SPEED: {sum(avg_speeds) / len(avg_speeds)}"
              f"\nAVG_FLOW: {sum(avg_flows) / len(avg_flows)}")

