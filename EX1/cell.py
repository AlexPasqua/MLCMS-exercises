class Cell:
    """
    Object to handle at a low level the graphical cell, its color filling and its status changing
    """

    EMPTY_COLOR_BG = "white"
    EMPTY_COLOR_BORDER = "black"
    FILLED_COLOR_BG = None
    FILLED_COLOR_BORDER = None

    def __init__(self, master, x, y, size):
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
            # print(self.borders[0], self.borders[1], self.borders[2], self.borders[3])
            self.master.create_rectangle(self.borders[0], self.borders[1], self.borders[2], self.borders[3], fill=fill,
                                         outline=outline)
        self.FILLED_COLOR_BG = fill
        self.FILLED_COLOR_BORDER = outline

    def update_status(self, fill):
        """
        given a filling color changes the state of the cell, updating also the cell_grid lists of current
        notable cells
        :param fill:
        """
        if fill in ('white'):
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
        elif fill == 'green':
            self.status = 'Target'
            self.master.targets_cell_list.append(self)
