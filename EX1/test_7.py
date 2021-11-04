import numpy as np
from tkinter import *
from grid import CellGrid
from pedestrian import Pedestrian


def test_7(num_pedestrians, path_len=10, screen_width=500):
    speed_list = sample_age_speed(num_pedestrians)
    app = Tk()
    grid = CellGrid(app, num_pedestrians, path_len, int(screen_width/num_pedestrians))
    grid.pack()
    grid.focus_set()  # to receive inputs form keyboard

    # draw person on first column
    for cell in range(num_pedestrians):
        set_person(grid, cell, 0)

    # draw target on last column
    for cell in range(num_pedestrians):
        set_target(grid, cell, path_len-1)

    # create Pedestrian list with custom speeds
    grid.pedestrian_list = [Pedestrian(grid, cell, speed_list[i]) for i,cell in enumerate(grid.pedestrian_cell_list)]
    app.mainloop()


def set_person(grid, row, col):
    grid.draw_person()
    cell = grid.grid[row][col]
    cell.switch()
    cell.draw(grid.FILLED_COLOR_BG, grid.FILLED_COLOR_BORDER)


def set_target(grid, row, col):
    grid.draw_target()
    cell = grid.grid[row][col]
    cell.switch()
    cell.draw(grid.FILLED_COLOR_BG, grid.FILLED_COLOR_BORDER)


def sample_age_speed(num_samples):
    """
    given a number of samples to be sampled, returns an array of speeds for the Pedestrians to create
    after sampling their age (going from 3 to 80 years)
    :param num_samples: number of samples of speed to take
    :return:
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


if __name__ == "__main__":
    test_7(10)