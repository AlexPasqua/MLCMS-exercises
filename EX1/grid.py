from tkinter import *
import time
import random
from planning_grid import PlanningGrid
from pedestrian import Pedestrian

"""
At the moment it is possible to edit the field of play as one prefers. Once the Run button is pressed the player can 
select one of the Person cells (red ones) and move it around using the arrow keys or wasd controls. It is possible to 
switch person just by clicking on a different one.
Player will win when reaching the target (green cells), and will vanish. Player cannot go through other players or 
obstacles.
"""


# TODO manage the edges when moving

class Cell:
    """
    Object to handle at a low level the graphical cell, its color filling and its status changing
    """

    EMPTY_COLOR_BG = "white"
    EMPTY_COLOR_BORDER = "black"

    def __init__(self, master, x, y, size):
        """ Constructor of the object called by Cell(...) """
        self.master = master
        self.abs = x
        self.ord = y
        self.size = size
        self.fill = False  # if False then the cell will be blanked out at next "draw" call
        self.borders = [self.abs * self.size,
                        self.ord * self.size,
                        (self.abs + 1) * self.size,
                        (self.ord + 1) * self.size]
        self.status = "Blank"  # 4 possible statuses: Blank, Person, Obstacle, Target

    def switch(self):
        """
        changes the fill modality: if fill is false then the next draw call will "Blank" the cell
        """
        self.fill = not self.fill

    def draw(self, fill, outline):
        """
        effectively draw the cell, changing its color
        :param fill:
        :param outline:
        """
        if self.master is not None:
            if not self.fill:
                fill = self.EMPTY_COLOR_BG
                outline = self.EMPTY_COLOR_BORDER

            self.update_status(fill)
            self.master.create_rectangle(self.borders[0], self.borders[1], self.borders[2], self.borders[3], fill=fill,
                                         outline=outline)

    def update_status(self, fill):
        """
        given a filling color changes the state of the cell, updating also the cell_grid lists of current
        notable cells
        :param fill:
        """
        if fill == 'white':
            # the removal from the lists is done only here since before changing color each cell has to return Blank
            if self.status == 'Person':
                self.master.pedestrian_cell_list.remove(self)
            elif self.status == 'Obstacle':
                self.master.obstacles_cell_list.remove(self)
            elif self.status == 'Target':
                self.master.targets_cell_list.remove(self)
            self.status = 'Blank'
        elif fill == 'red':
            self.status = 'Person'
            self.master.pedestrian_cell_list.append(self)
        elif fill == 'black':
            self.status = 'Obstacle'
            self.master.obstacles_cell_list.append(self)
        else:
            self.status = 'Target'
            self.master.targets_cell_list.append(self)


def close_app(event):
    """
    press the ESCAPE key to terminate the program
    :param event:
    :return:
    """
    sys.exit()


class CellGrid(Canvas):
    """
    Object to handle the overall execution, maintaining the graphical grid and switching between different modalities
    """
    def __init__(self, master, row_number, column_number, cell_size, *args, **kwargs):
        self.canvas = Canvas.__init__(self, master, width=cell_size * column_number, height=cell_size * row_number,
                                      *args, **kwargs)
        self.master = master
        self.FILLED_COLOR_BG = "green"
        self.FILLED_COLOR_BORDER = "green"
        self.TIME_STEP = 0.2  # needed for simulation
        self.selected_pedestrian = None  # needed for free walk mode
        self.targets_cell_list = []
        self.pedestrian_cell_list = []  # lower level structure, containing only the Cell object
        self.pedestrian_list = []       # higher level structure, containing Pedestrian object
        self.obstacles_cell_list = []
        self.cellSize = cell_size

        # buttons and their initialization
        self.person_button = None
        self.obstacle_button = None
        self.target_button = None
        self.run_simulation_button = None
        self.free_walk_button = None
        self.buttons = []
        self.init_buttons()

        # graphical grid init
        self.grid = []
        for row in range(row_number):
            line = [Cell(self, column, row, cell_size) for column in range(column_number)]
            self.grid.append(line)

        # memorize the cells that have been modified to avoid many switching of state during mouse motion.
        self.switched = []

        # bind click action
        self.bind("<Button-1>", self.handle_mouse_click)
        # bind moving while clicking
        self.bind("<B1-Motion>", self.handle_mouse_motion)
        # bind release button action - clear the memory of modified cells.
        self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())
        # bind escape button action - exit application
        self.bind("<Escape>", close_app)

        # draw grid
        self.draw_empty_grid()

    def draw_empty_grid(self):
        for row in self.grid:
            for cell in row:
                cell.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)

    def _event_coords(self, event):
        """
        to capture the correct locations of keyboard/mouse triggered events
        :param event:
        :return:
        """
        row = int(event.y / self.cellSize)
        column = int(event.x / self.cellSize)
        return row, column

    def handle_mouse_click(self, event):
        """
        function to handle left mouse click, coloring a specific cell
        :param event:
        :return:
        """

        row, column = self._event_coords(event)
        cell = self.grid[row][column]
        cell.switch()
        cell.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)
        # add the cell to the list of cell switched during the click
        self.switched.append(cell)

    def handle_mouse_motion(self, event):
        """
        function to handle the multiple coloring feature, happening when clicking and maintaining left mouse button
        :param event:
        :return:
        """
        row, column = self._event_coords(event)
        cell = self.grid[row][column]
        if cell not in self.switched:
            cell.switch()
            cell.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)
            self.switched.append(cell)

    def init_buttons(self):
        """
        function to initialize all buttons, binding them to a certain function to trigger when an event arrives
        :return:
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

    def update_buttons_state(self, current_button_text):
        """
        rises the previously pressed button, keeps pressed the last fired button
        :param current_button_text:
        :return:
        """
        for button in self.buttons:
            if button['text'] == current_button_text:
                button.configure(relief=SUNKEN, state=DISABLED)
            elif button['state'] == DISABLED:
                button.configure(relief=RAISED, state=ACTIVE)

    def draw_person(self):
        """
        prepares the color configuration for drawing Person/Pedestrian
        :return:
        """
        self.update_buttons_state("Person")
        color = "red"
        self.FILLED_COLOR_BG = color
        self.FILLED_COLOR_BORDER = color

    def draw_obstacle(self):
        """
        prepares the color configuration for drawing Obstacle
        :return:
        """
        self.update_buttons_state("Obstacle")
        color = "black"
        self.FILLED_COLOR_BG = color
        self.FILLED_COLOR_BORDER = color

    def draw_target(self):
        """
        prepares the color configuration for drawing Target
        :return:
        """
        self.update_buttons_state("Target")
        color = "green"
        self.FILLED_COLOR_BG = color
        self.FILLED_COLOR_BORDER = color

    def switch_mode(self):
        """
        handler for switching back and forth between Editing Mode and Free Walk Mode
        :return:
        """
        if self.free_walk_button['text'] == 'Free Walk':
            self.free_walk_mode()
        else:
            self.editing_mode()

    def free_walk_mode(self):
        """
        changes the key bindings for handling the free walk mode
        :return:
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
        changes the key bindings for handling the editing mode
        :return:
        """
        print('Entering movement mode..')
        self.bind("<Button-1>", self.handle_mouse_click)
        self.bind("<B1-Motion>", self.handle_mouse_motion)
        self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())
        self.unbind(("<Left>", "<Right>", "<Up>", "<Down>", "a", "s", "d", "w"))
        self.person_button.configure(relief=RAISED, state=ACTIVE)  # person
        self.obstacle_button.configure(relief=RAISED, state=ACTIVE)  # obstacle
        self.target_button.configure(relief=SUNKEN, state=DISABLED)  # target
        self.draw_target()
        self.free_walk_button['text'] = "Free Walk"  # free walk / editing mode

    def select_person(self, event):
        """
        left-clicking on a Person cell will select it, making it able to be moved
        :param event:
        :return:
        """
        row, column = self._event_coords(event)
        candidate_cell = self.grid[row][column]
        if candidate_cell.status == 'Person':
            self.selected_pedestrian = candidate_cell

    def move_person(self, event=None, movement=None):
        """
        function to handle movement in FREE WALK MODE
        :param event:
        :param movement:
        :return:
        """
        # TODO need to check boundaries!
        if self.selected_pedestrian is not None:  # a pedestrian has to be selected by left-clicking it
            self.selected_pedestrian.switch()
            self.selected_pedestrian.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)
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
        :return:
        """
        print('Entering simulation mode..')
        # unbind unnecessary keys
        self.unbind("<Button-1>")
        self.unbind("<B1-Motion>")
        self.unbind("<ButtonRelease-1>")
        self.FILLED_COLOR_BORDER = 'red'
        self.FILLED_COLOR_BG = 'red'
        self.person_button.configure(relief=SUNKEN, state=DISABLED)
        self.obstacle_button.configure(relief=SUNKEN, state=DISABLED)
        self.target_button.configure(relief=SUNKEN, state=DISABLED)
        self.free_walk_button['text'] = "Editing mode"

        # create a PlanningGrid for more efficient management of simulation
        planning_grid = PlanningGrid(self)

        # create a Pedestrian list to access useful methods
        self.pedestrian_list = [Pedestrian(self, cell) for cell in self.pedestrian_cell_list]

        # continue simulating until all pedestrian have not reached a target
        while len(self.pedestrian_list) != 0:
            random.shuffle(self.pedestrian_list)  # to avoid giving advantage to the same Pedestrian all the time
            for pedestrian in self.pedestrian_list:
                pedestrian.update_cost_function(planning_grid)  # update local cost_matrix
                planning_grid, pedestrian_has_ended = pedestrian.move()  # try to move (time constraints)
                if pedestrian_has_ended:
                    self.pedestrian_list.remove(pedestrian)

            time.sleep(self.TIME_STEP)  # problem discretization
            self.update()   # graphical update of the grid


if __name__ == "__main__":
    app = Tk()
    grid = CellGrid(app,  30, 30, 20)
    grid.pack()
    grid.focus_set()  # to receive inputs form keyboard
    app.mainloop()
