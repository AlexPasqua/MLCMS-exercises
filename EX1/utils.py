import random
import numpy as np
from tkinter import *
from grid import CellGrid
from pedestrian import Pedestrian


def set_person(grid, row, col):
    """ Set a pedestrian in a specific cell in a specific grid """
    grid.draw_person()
    cell = grid.grid[row][col]
    cell.switch()
    cell.draw(grid.FILLED_COLOR_BG, grid.FILLED_COLOR_BORDER)


def set_target(grid, row, col):
    """ Set a target in a specific cell in a specific grid"""
    grid.draw_target()
    cell = grid.grid[row][col]
    cell.switch()
    cell.draw(grid.FILLED_COLOR_BG, grid.FILLED_COLOR_BORDER)


def set_obstacle(grid, row, col):
    """ Set an obstacle in a specific cell in a specific grid"""
    grid.draw_obstacle()
    cell = grid.grid[row][col]
    cell.switch()
    cell.draw(grid.FILLED_COLOR_BG, grid.FILLED_COLOR_BORDER)


def setup_task_1(app):
    """ Create environment for task 1 """
    grid = CellGrid(app, 50, 10)
    grid.pack()
    grid.focus_set()  # to receive inputs form keyboard


def setup_task_2(app):
    """ Create environment for task 2 """
    grid = CellGrid(app, 50, 10)
    grid.pack()
    grid.focus_set()  # to receive inputs form keyboard
    # draw person at (5,25)
    set_person(grid, 4, 24)
    # draw target at (25,25)
    set_target(grid, 24, 24)


def setup_task_3(app):
    """ Create environment for task 3 """
    grid = CellGrid(app, 50, 10)
    grid.pack()
    grid.focus_set()  # to receive inputs form keyboard
    # draw person at (10,10)
    set_person(grid, 9, 9)
    # draw person at (4,25)
    set_person(grid, 3, 24)
    # draw person at (46,25)
    set_person(grid, 45, 24)
    # draw person at (25,46)
    set_person(grid, 24, 45)
    # draw person at (20,6)
    set_person(grid, 19, 5)
    # draw target at (25,25)
    set_target(grid, 24, 24)


def setup_task_4(app):
    """ Create environment for task 4 """
    grid = CellGrid(app, 10, 50)
    grid.pack()
    set_target(grid, 0, 4)
    set_person(grid, 6, 4)
    for i in range(2, 7):
        set_obstacle(grid, 2, i)
    for i in range(3, 5):
        set_obstacle(grid, i, 2)
        set_obstacle(grid, i, 6)


def rimea_test_1(grid_size=40, screen_width=500):
    """ Create environment for RiMEA scenario 1 in task 5 """
    app = Tk()
    grid = CellGrid(app, grid_size, int(screen_width / grid_size))
    grid.pack()
    grid.focus_set()  # to receive inputs form keyboard

    # draw corridor (two lines of obstacles)
    for i in range(grid_size):
        set_obstacle(grid, (grid_size // 2) - 1, i)
        set_obstacle(grid, (grid_size // 2) + 2, i)

    # draw person at the beginning of corridor
    set_person(grid, (grid_size // 2), 0)

    # draw target at the end of the corridor
    set_target(grid, (grid_size // 2), grid_size - 1)
    set_target(grid, (grid_size // 2) + 1, grid_size - 1)

    app.mainloop()


def rimea_test_4(screen_width=500):
    """ Create environment for RiMEA scenario 4 in task 5 """
    grid_size = 80
    app = Tk()
    grid = CellGrid(app, grid_size, int(screen_width / grid_size), is_rimea_4=True)
    grid.pack()
    grid.focus_set()  # to receive inputs form keyboard

    # draw corridor (two lines of obstacles)
    corridor_width = 5
    for i in range(grid_size):
        set_obstacle(grid, (grid_size // 2) - corridor_width, i)
        set_obstacle(grid, (grid_size // 2) + corridor_width, i)

    # draw pedestrians at the beginning of corridor (specifying the spawning area)
    min_col = 0
    max_col = grid_size - 4
    min_row = (grid_size // 2) - corridor_width + 1
    max_row = (grid_size // 2) + corridor_width - 1
    possible_cells = []
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            possible_cells.append((i, j))
    num_pedestrians = len(possible_cells) // 2
    print(f"PRODUCING {num_pedestrians} PEDESTRIANS IN AN AREA OF {len(possible_cells)} CELLS")
    while num_pedestrians > 0:
        candidate_cell = random.choice(possible_cells)
        possible_cells.remove(candidate_cell)
        set_person(grid, candidate_cell[0], candidate_cell[1])
        num_pedestrians -= 1
    set_person(grid, 39, 13)

    # draw target at the end of the corridor
    for i in range((grid_size // 2) - corridor_width + 1, (grid_size // 2) + corridor_width):
        set_target(grid, i, grid_size - 1)

    app.mainloop()


def rimea_test_6(screen_width=500):
    """ Create environment for RiMEA scenario 6 in task 5 """
    grid_size = 13
    num_pedestrians = 20
    app = Tk()
    grid = CellGrid(app, grid_size, int(screen_width / grid_size))
    grid.pack()
    grid.focus_set()  # to receive inputs form keyboard

    # draw corner (two lines of obstacles)
    # these lines looks rather complicated, but we could have simply hardcoded the coordinates of the obstacles,
    # this way it was a little easier to change the disposition and shape of the corner during testing.
    # Anyway this simply places 2 L-shaped lines of obstacles to create a corridor with a 90° turn as the one present
    # in RiMEA scenario 6
    for i in range(grid_size):
        if i in range(0, 8):
            set_obstacle(grid, i, grid_size - 1)
            set_obstacle(grid, i, grid_size - 4)
        elif i == 8:
            for j in range(grid_size):
                if j not in (grid_size - 2, grid_size - 3):
                    set_obstacle(grid, i, j)
        elif i in (9, 10, 11):
            set_obstacle(grid, i, grid_size - 1)
        else:
            for j in range(grid_size):
                set_obstacle(grid, i, j)

    # draw targets
    set_target(grid, 0, grid_size - 2)
    set_target(grid, 0, grid_size - 3)

    # pick pedestrians at random
    min_col, max_col, min_row, max_row = 0, 8, grid_size - 4, grid_size - 2  # boundaries for pedestrian spawning area
    possible_cells = []
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            possible_cells.append((i, j))
    while num_pedestrians > 0:
        candidate_cell = random.choice(possible_cells)
        possible_cells.remove(candidate_cell)
        set_person(grid, candidate_cell[0], candidate_cell[1])
        num_pedestrians -= 1

    app.mainloop()


def rimea_test_7(num_pedestrians=50, screen_width=500):
    """ Create environment for RiMEA scenario 7 in task 5 """
    speed_list = sample_age_speed(num_pedestrians)
    app = Tk()
    grid = CellGrid(app, num_pedestrians, int(screen_width / num_pedestrians))
    grid.pack()
    grid.focus_set()  # to receive inputs form keyboard

    # draw person on first column
    for cell in range(num_pedestrians):
        set_person(grid, cell, 0)

    # draw target on last column
    for cell in range(num_pedestrians):
        set_target(grid, cell, num_pedestrians - 1)

    # create Pedestrian list with custom speeds
    grid.pedestrian_list = [
        Pedestrian(grid, cell, speed_list[i], num_pedestrians >= 100) for i, cell in enumerate(grid.pedestrian_cell_list)
    ]
    app.mainloop()
    print("Measured Average Speed: ", sum(grid.pedestrian_speeds) / len(grid.pedestrian_speeds))
    print("Expected Average Speed: ", sum(speed_list) / len(speed_list))


def sample_age_speed(num_samples):
    """
    given a number of samples to be sampled, returns an array of speeds for the Pedestrians to create
    after sampling their age (going from 3 to 80 years)
    :param num_samples: number of samples of speed to take
    :return: the list containing the speeds (rounded to the 2nd decimal)
    """
    lowest_age, highest_age = 3, 80
    first_range_speed = [0.6, 1.2]  # 3-10 years
    second_range_speed = [1.2, 1.6]  # 11-20 years
    third_range_speed = [1.6, 1.4]  # 21-50 years
    fourth_range_speed = [1.4, 1.1]  # 51-70 years
    fifth_range_speed = [1.1, 0.7]  # 71-80 years
    sampled_ages = np.random.randint(lowest_age, highest_age, num_samples)

    speed_list = []
    for age in sampled_ages:
        if 3 <= age <= 10:
            speed_list.append(np.random.uniform(first_range_speed[0], first_range_speed[1]))
        elif 11 <= age <= 20:
            speed_list.append(np.random.uniform(second_range_speed[0], second_range_speed[1]))
        elif 21 <= age <= 50:
            speed_list.append(np.random.uniform(third_range_speed[0], third_range_speed[1]))
        elif 51 <= age <= 70:
            speed_list.append(np.random.uniform(fourth_range_speed[0], fourth_range_speed[1]))
        else:
            speed_list.append(np.random.uniform(fifth_range_speed[0], fifth_range_speed[1]))
    return [round(speed, 2) for speed in speed_list]
