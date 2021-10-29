import copy
from tkinter import *
import math
import time
import numpy as np
import random

"""
At the moment it is possible to edit the field of play as one prefers. Once the Run button is pressed the player can 
select one of the Person cells (red ones) and move it around using the arrow keys or wasd controls. It is possible to 
switch person just by clicking on a different one.
Player will win when reaching the target (green cells), and will vanish. Player cannot go through other players or 
obstacles.
"""


# TODO manage the edges when moving

class Cell:
    EMPTY_COLOR_BG = "white"
    EMPTY_COLOR_BORDER = "black"

    def __init__(self, master, x, y, size):
        """ Constructor of the object called by Cell(...) """
        self.master = master
        self.abs = x
        self.ord = y
        self.size = size
        self.fill = False
        self.status = "Blank"
        self.cost = 0
        self.borders = [self.abs * self.size,
                        self.ord * self.size,
                        (self.abs + 1) * self.size,
                        (self.ord + 1) * self.size]

    def switch(self):
        """
        changes the fill modality: if fill is false then the next draw call will "Blank" the cell
        :return:
        """
        self.fill = not self.fill

    def draw(self, fill, outline):
        """
        effectively draw the cell, changing its color
        :param fill:
        :param outline:
        :return:
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
        :return:
        """
        if fill == 'white':
            # the removal from the lists is done only here since before changing color each cell has to return Blank
            if self.status == 'Person':
                self.master.pedestrian_list.remove(self)
            elif self.status == 'Obstacle':
                self.master.obstacle_list.remove(self)
            elif self.status == 'Target':
                self.master.target_list.remove(self)
            self.status = 'Blank'
        elif fill == 'red':
            self.status = 'Person'
            self.master.pedestrian_list.append(self)
        elif fill == 'black':
            self.status = 'Obstacle'
            self.master.obstacle_list.append(self)
        else:
            self.status = 'Target'
            self.master.target_list.append(self)


def close_app(event):
    sys.exit()


class CellGrid(Canvas):
    def __init__(self, master, row_number, column_number, cell_size, *args, **kwargs):
        self.canvas = Canvas.__init__(self, master, width=cell_size * column_number, height=cell_size * row_number,
                                      *args, **kwargs)
        self.FILLED_COLOR_BG = "green"
        self.FILLED_COLOR_BORDER = "green"
        self.selected_pedestrian = None  # needed after editing
        self.target_list = []
        self.pedestrian_list = []
        self.obstacle_list = []
        self.person_button = None
        self.obstacle_button = None
        self.target_button = None
        self.run_simulation_button = None
        self.free_walk_button = None
        self.buttons = []
        self.init_buttons()

        self.cellSize = cell_size

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

        self.draw_empty_grid()

    def draw_empty_grid(self):
        for row in self.grid:
            for cell in row:
                cell.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)

    def _event_coords(self, event):
        row = int(event.y / self.cellSize)
        column = int(event.x / self.cellSize)
        return row, column

    def handle_mouse_click(self, event):
        row, column = self._event_coords(event)
        cell = self.grid[row][column]
        cell.switch()
        cell.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)
        # add the cell to the list of cell switched during the click
        self.switched.append(cell)

    def handle_mouse_motion(self, event):
        row, column = self._event_coords(event)
        cell = self.grid[row][column]
        if cell not in self.switched:
            cell.switch()
            cell.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)
            self.switched.append(cell)

    def init_buttons(self):
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
        self.target_button.configure(relief=SUNKEN, state=DISABLED)

    def update_buttons_state(self, current_button_text):
        for button in self.buttons:
            if button['text'] == current_button_text:
                button.configure(relief=SUNKEN, state=DISABLED)
            elif button['state'] == DISABLED:
                button.configure(relief=RAISED, state=ACTIVE)

    def draw_person(self):
        self.update_buttons_state("Person")
        color = "red"
        self.FILLED_COLOR_BG = color
        self.FILLED_COLOR_BORDER = color

    def draw_obstacle(self):
        self.update_buttons_state("Obstacle")
        color = "black"
        self.FILLED_COLOR_BG = color
        self.FILLED_COLOR_BORDER = color

    def draw_target(self):
        self.update_buttons_state("Target")
        color = "green"
        self.FILLED_COLOR_BG = color
        self.FILLED_COLOR_BORDER = color

    def switch_mode(self):
        """
        enter free walk mode -> override key bindings to control certain person
        :return:
        """
        if self.free_walk_button['text'] == 'Free Walk':
            self.free_walk_mode()
        else:
            self.editing_mode()

    def free_walk_mode(self):
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
        row, column = self._event_coords(event)
        candidate_cell = self.grid[row][column]
        print(candidate_cell.status)
        if candidate_cell.status == 'Person':
            self.selected_pedestrian = candidate_cell

    def move_person(self, event=None, movement=None):
        # TODO need to check boundaries!
        if self.selected_pedestrian is not None:
            self.selected_pedestrian.switch()
            self.selected_pedestrian.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)
            if event is not None:
                if event.keysym == 'Right' or event.keysym == 'd':
                    candidate_cell = self.grid[self.selected_pedestrian.ord][self.selected_pedestrian.abs + 1]
                elif event.keysym == 'Left' or event.keysym == 'a':
                    candidate_cell = self.grid[self.selected_pedestrian.ord][self.selected_pedestrian.abs - 1]
                elif event.keysym == 'Up' or event.keysym == 'w':
                    candidate_cell = self.grid[self.selected_pedestrian.ord - 1][self.selected_pedestrian.abs]
                else:
                    candidate_cell = self.grid[self.selected_pedestrian.ord + 1][self.selected_pedestrian.abs]
            elif movement is not None:
                if movement == '<<Up_Left>>':
                    candidate_cell = self.grid[self.selected_pedestrian.ord - 1][self.selected_pedestrian.abs - 1]
                elif movement == '<<Up>>':
                    candidate_cell = self.grid[self.selected_pedestrian.ord - 1][self.selected_pedestrian.abs]
                elif movement == '<<Up_Right>>':
                    candidate_cell = self.grid[self.selected_pedestrian.ord - 1][self.selected_pedestrian.abs + 1]
                elif movement == '<<Left>>':
                    candidate_cell = self.grid[self.selected_pedestrian.ord][self.selected_pedestrian.abs - 1]
                elif movement == '<<Right>>':
                    candidate_cell = self.grid[self.selected_pedestrian.ord][self.selected_pedestrian.abs + 1]
                elif movement == '<<Down_Left>>':
                    candidate_cell = self.grid[self.selected_pedestrian.ord + 1][self.selected_pedestrian.abs - 1]
                elif movement == '<<Down>>':
                    candidate_cell = self.grid[self.selected_pedestrian.ord + 1][self.selected_pedestrian.abs]
                else:
                    candidate_cell = self.grid[self.selected_pedestrian.ord + 1][self.selected_pedestrian.abs + 1]
            else:
                print("Something here is fishy..")
            if candidate_cell.status not in ('Obstacle', 'Person'):
                self.selected_pedestrian = candidate_cell
            if candidate_cell.status == 'Target':
                self.selected_pedestrian = None
            else:
                self.selected_pedestrian.switch()
                self.selected_pedestrian.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)

    def start_simulation(self):
        [cell.master.delete('cost_value') for line in self.grid for cell in line]
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
        while len(self.pedestrian_list) != 0:
            temp_pedestrian_list = self.pedestrian_list
            for pedestrian in temp_pedestrian_list:
                print(pedestrian)
                self.update_cost_function()
                self.selected_pedestrian = pedestrian
                self.next_movement(pedestrian)
                self.update()

            #pseudo code
            random.shuffle(self.pedestrian_list)
            temp_grid = copy.deepcopy(self)
            for pedestrian in self.pedestrian_list:
                pedestrian.update_cost_function(temp_grid)
                pedestrian.move(temp_grid) # check if active, if not pass. if active then plan the move, set timestamp, set to sleeping
            self.advance_time() # check all sleeping, diminish time by 0.2, if time remaining = 0.0 append to moving_people, move them, update, make them active






            time.sleep(1)

    def next_movement(self, pedestrian):
        """
        check 8 cells surrounding pedestrian, understanding where the lowest cost is.
        A commodity array is constructed filling 8 positions so to have an easier way to get the final direction
        :param pedestrian:
        :return:
        """
        print("Calculating next move for pedestrian:", pedestrian.ord, pedestrian.abs, end=" - ")
        surrounding_costs = np.zeros(8)
        surrounding_costs[0] += self.grid[pedestrian.ord - 1][pedestrian.abs - 1].cost \
            if pedestrian.ord - 1 >= 0 and pedestrian.abs - 1 >= 0 else math.inf  # upper-left neighbour

        surrounding_costs[1] += self.grid[pedestrian.ord - 1][pedestrian.abs].cost \
            if pedestrian.ord - 1 >= 0 else math.inf  # upper-mid neighbour

        surrounding_costs[2] += self.grid[pedestrian.ord - 1][pedestrian.abs + 1].cost \
            if pedestrian.ord - 1 >= 0 and pedestrian.abs + 1 < len(self.grid[0]) else math.inf  # upper-right neighbour

        surrounding_costs[3] += self.grid[pedestrian.ord][pedestrian.abs - 1].cost \
            if pedestrian.abs - 1 >= 0 else math.inf  # mid-left neighbour

        surrounding_costs[4] += self.grid[pedestrian.ord][pedestrian.abs + 1].cost \
            if pedestrian.abs + 1 < len(self.grid[0]) else math.inf  # mid-right neighbour

        surrounding_costs[5] += self.grid[pedestrian.ord + 1][pedestrian.abs - 1].cost \
            if pedestrian.ord + 1 < len(self.grid) and pedestrian.abs - 1 >= 0 else math.inf  # lower-left neighbour

        surrounding_costs[6] += self.grid[pedestrian.ord + 1][pedestrian.abs].cost \
            if pedestrian.ord + 1 < len(self.grid) else math.inf  # lower-mid neighbour

        surrounding_costs[7] += self.grid[pedestrian.ord + 1][pedestrian.abs + 1].cost \
            if pedestrian.ord + 1 < len(self.grid) and pedestrian.abs + 1 < len(
            self.grid[0]) else math.inf  # lower-right neighbour
        selected_movement = ["<<Up_Left>>", "<<Up>>", "<<Up_Right>>", "<<Left>>",
                             "<<Right>>", "<<Down_Left>>", "<<Down>>", "<<Down_Right>>"][np.argmin(surrounding_costs)]
        print("Calculated motion:", selected_movement)
        self.move_person(movement=selected_movement)

    def update_cost_function(self):
        # put to zero otherwise would continue increasing with further updates
        for line in self.grid:
            for cell in line:
                cell.cost = 0

        # for each target add distance cost to cells
        for target in self.target_list:
            target_pos = np.array([target.ord, target.abs])
            for line in self.grid:
                for cell in line:
                    cell_pos = np.array([cell.ord, cell.abs])
                    cell.cost += np.linalg.norm(target_pos - cell_pos)

        # targets cost 0
        for target in self.target_list:
            target.cost = 0

        # pedestrians and obstacles are unreachable -> infinite cost
        for pedestrian in self.pedestrian_list:
            pedestrian.cost = math.inf
        for obstacle in self.obstacle_list:
            obstacle.cost = math.inf

        # print cost on cell (at the moment)
        # TODO get rid of this
        for line in self.grid:
            for cell in line:
                cell.master.create_text(((cell.borders[0] + cell.borders[2]) // 2,
                                         (cell.borders[1] + cell.borders[3]) // 2),
                                        text=round(cell.cost, 1), tags='cost_value')


if __name__ == "__main__":
    app = Tk()
    grid = CellGrid(app, 5, 5, 50)
    grid.pack()
    grid.focus_set()  # to receive inputs form keyboard
    app.mainloop()
