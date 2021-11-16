import os

import task3.modify_scenario as modify_scenario


if __name__ == "__main__":
    """
        Automatic creation of Corridor Scenario needed for task 4.5
        It assumes that the corridor is of 40x20 with a target id of 2
    """
    num_pedestrian_to_generate = 100
    scenario_to_modify = os.getcwd() + "\\task4.5\\scenarios\\task4.5_2.scenario"
    for i in range(num_pedestrian_to_generate):
        modify_scenario.add_pedestrian(
            id=num_pedestrian_to_generate+i,
            scenario_path=scenario_to_modify,
            out_scen_name="task4.5_2",
            output_path=scenario_to_modify,
            position=(i // 20, i % 20),
            targetIds=[2],
            groupIds=[0] # infected
        )