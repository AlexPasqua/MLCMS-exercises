from tkinter import *

"""
At the moment it is possible to edit the field of play as one prefers. Once the Run button is pressed the player can 
select one of the Person cells (red ones) and move it around using the arrow keys or wasd controls. It is possible to 
switch person just by clicking on a different one.
Player will win when reaching the target (green cells), and will vanish. Player cannot go through other players or 
obstacles.
"""


# TODO manage the edges
# TODO make the player another color so that it is recognizable

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

    def switch(self):
        """ Switch if the cell is filled or not. """
        self.fill = not self.fill

    def draw(self, fill, outline):
        """ order to the cell to draw_empty_grid its representation on the canvas """
        if self.master is not None:
            if not self.fill:
                fill = self.EMPTY_COLOR_BG
                outline = self.EMPTY_COLOR_BORDER

            x_min = self.abs * self.size
            x_max = x_min + self.size
            y_min = self.ord * self.size
            y_max = y_min + self.size

            self.update_status(fill)
            self.master.create_rectangle(x_min, y_min, x_max, y_max, fill=fill, outline=outline)

    def update_status(self, fill):
        if fill == 'white':
            self.status = 'Blank'
        elif fill == 'red':
            self.status = 'Person'
        elif fill == 'black':
            self.status = 'Obstacle'
        else:
            self.status = 'Target'


def close_app(event):
    sys.exit()


class CellGrid(Canvas):
    def __init__(self, master, row_number, column_number, cell_size, *args, **kwargs):
        self.canvas = Canvas.__init__(self, master, width=cell_size * column_number, height=cell_size * row_number,
                                      *args, **kwargs)
        self.FILLED_COLOR_BG = "green"
        self.FILLED_COLOR_BORDER = "green"
        self.selected_pedestrian = None  # needed after editing
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
        person_button = Button(self.canvas, text="Person", command=self.draw_person)
        person_button.pack()
        obstacle_button = Button(self.canvas, text="Obstacle", command=self.draw_obstacle)
        obstacle_button.pack()
        target_button = Button(self.canvas, text="Target", command=self.draw_target)
        target_button.pack()
        run_button = Button(self.canvas, text="Run", command=self.switch_mode)
        run_button.pack()
        self.buttons = [person_button, obstacle_button, target_button, run_button]
        target_button.configure(relief=SUNKEN, state=DISABLED)

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
        enter run mode -> override key bindings to control certain person
        :return:
        """
        if self.buttons[-1]['text'] == 'Run':
            self.run_mode()
        else:
            self.editing_mode()

    def run_mode(self):
        print('Entering movement mode..')
        # bind click action
        self.bind("<Button-1>", self.select_person)
        # bind left arrow and 'a' action
        self.bind("<Left>", self.move_person) and self.bind("a", self.move_person)
        # bind right arrow and 'd' action
        self.bind("<Right>", self.move_person) and self.bind("d", self.move_person)
        # bind up arrow and 'w' action
        self.bind("<Up>", self.move_person) and self.bind("w", self.move_person)
        # bind down arrow and 's' action
        self.bind("<Down>", self.move_person) and self.bind("s", self.move_person)
        # unbind unnecessary keys
        self.unbind("<B1-Motion>")
        self.unbind("<ButtonRelease-1>")
        self.FILLED_COLOR_BORDER = 'red'
        self.FILLED_COLOR_BG = 'red'
        for button in self.buttons[: -1]:
            button.configure(relief=SUNKEN, state=DISABLED)
        self.buttons[-1]['text'] = "Editing mode"

    def editing_mode(self):
        self.bind("<Button-1>", self.handle_mouse_click)
        self.bind("<B1-Motion>", self.handle_mouse_motion)
        self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())
        self.unbind(("<Left>", "<Right>", "<Up>", "<Down>", "a", "s", "d", "w"))
        for button in self.buttons[: -1]:
            button.configure(relief=RAISED, state=ACTIVE)
        self.buttons[0].configure(relief=RAISED, state=ACTIVE)  # person
        self.buttons[1].configure(relief=RAISED, state=ACTIVE)  # obstacle
        self.buttons[2].configure(relief=SUNKEN, state=DISABLED)  # target
        self.draw_target()
        self.buttons[3]['text'] = "Run"     # run / editing mode

    def select_person(self, event):
        row, column = self._event_coords(event)
        candidate_cell = self.grid[row][column]
        print(candidate_cell.status)
        if candidate_cell.status == 'Person':
            self.selected_pedestrian = candidate_cell

    def move_person(self, event):
        if self.selected_pedestrian is not None:
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
            if candidate_cell.status not in ('Obstacle', 'Person'):
                self.selected_pedestrian = candidate_cell
            if candidate_cell.status == 'Target':
                self.selected_pedestrian = None
            else:
                self.selected_pedestrian.switch()
                self.selected_pedestrian.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)


if __name__ == "__main__":
    app = Tk()
    grid = CellGrid(app, 50, 50, 10)
    grid.pack()
    grid.focus_set()  # to receive inputs form keyboard
    app.mainloop()
