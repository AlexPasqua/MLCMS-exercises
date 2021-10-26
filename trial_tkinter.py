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
        """ order to the cell to draw its representation on the canvas """
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
        self.status = 'Blank' if fill == 'white' else (
            'Person' if fill == 'red' else ('Obstacle' if fill == 'black' else 'Target'))


def close_app(event):
    sys.exit()


class CellGrid(Canvas):
    FILLED_COLOR_BG = "green"
    FILLED_COLOR_BORDER = "green"

    def __init__(self, master, row_number, column_number, cell_size, *args, **kwargs):
        self.canvas = Canvas.__init__(self, master, width=cell_size * column_number, height=cell_size * row_number,
                                      *args, **kwargs)
        self.selected_cell = None  # needed after editing
        self.buttons = []
        self.init_buttons()

        self.cellSize = cell_size

        self.grid = []
        for row in range(row_number):

            line = []
            for column in range(column_number):
                line.append(Cell(self, column, row, cell_size))

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

        self.draw()

    def draw(self):
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
        run_button = Button(self.canvas, text="Run", command=self.end_editing)
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
        self.FILLED_COLOR_BG = color if self.FILLED_COLOR_BG != color else "green"
        self.FILLED_COLOR_BORDER = color if self.FILLED_COLOR_BORDER != color else "green"

    def draw_obstacle(self):
        self.update_buttons_state("Obstacle")
        color = "black"
        self.FILLED_COLOR_BG = color if self.FILLED_COLOR_BG != color else "green"
        self.FILLED_COLOR_BORDER = color if self.FILLED_COLOR_BORDER != color else "green"

    def draw_target(self):
        self.update_buttons_state("Target")
        color = "green"
        self.FILLED_COLOR_BG = color if self.FILLED_COLOR_BG != color else "green"
        self.FILLED_COLOR_BORDER = color if self.FILLED_COLOR_BORDER != color else "green"

    def end_editing(self):
        """
        enter run mode -> override key bindings to control certain person
        :return:
        """
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

    def select_person(self, event):
        row, column = self._event_coords(event)
        candidate_cell = self.grid[row][column]
        print(candidate_cell.status)
        if candidate_cell.status == 'Person':
            self.selected_cell = candidate_cell

    def move_person(self, event):
        self.selected_cell.switch()
        self.selected_cell.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)
        if event.keysym == 'Right' or event.keysym == 'd':
            candidate_cell = self.grid[self.selected_cell.ord][self.selected_cell.abs + 1]
        elif event.keysym == 'Left' or event.keysym == 'a':
            candidate_cell = self.grid[self.selected_cell.ord][self.selected_cell.abs - 1]
        elif event.keysym == 'Up' or event.keysym == 'w':
            candidate_cell = self.grid[self.selected_cell.ord - 1][self.selected_cell.abs]
        else:
            candidate_cell = self.grid[self.selected_cell.ord + 1][self.selected_cell.abs]
        if candidate_cell.status not in ('Obstacle', 'Person'):
            self.selected_cell = candidate_cell
        if candidate_cell.status == 'Target':
            self.selected_cell = None
        self.selected_cell.switch()
        self.selected_cell.draw(self.FILLED_COLOR_BG, self.FILLED_COLOR_BORDER)


if __name__ == "__main__":
    app = Tk()

    grid = CellGrid(app, 50, 50, 10)
    grid.pack()
    grid.focus_set()

    app.mainloop()
