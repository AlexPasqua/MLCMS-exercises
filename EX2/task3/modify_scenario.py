import json


def read_scenario(path="scenarios/rimea6.scenario"):
    """
    Loads a scenario file with json
    :param path: the path of the scenario file
    :return: the dictionary containing the scenario's data
    """
    with open(path, 'r') as f:
        scenario = json.load(f)
        return scenario


def add_pedestrian(scenario):
    ped = {}
    id = find_free_id(scenario)
    print(id)


def find_free_id(scenario: dict, find_min_free_id=True):
    """
    Find a free id for a new pedestrian/target
    :param scenario: dictionary containing a scenario's data
    :param find_min_free_id: if True, finds the minimum free id (less efficient), otherwise simply a free id (more efficient)
    :return: a free id (int)
    """
    busy_ids = set()
    # iterate over pedestrians to collect their (already busy) ids
    dynamic_elems = scenario['scenario']['topography']['dynamicElements']
    for elem in dynamic_elems:
        if elem['type'] == 'PEDESTRIAN':
            busy_ids.add(elem['attributes']['id'])

    # iterate over targets to collect their (already busy) ids
    targets = scenario['scenario']['topography']['targets']
    for t in targets:
        busy_ids.add(t['id'])

    if not find_min_free_id:
        return max(busy_ids) + 1    # simply return the max busy id + 1 (which will be free)

    # otherwise sort the busy ids and find the minimum free one
    sorted_ids = sorted(list(busy_ids))
    try:
        # in case sorted_ids is empty, this would cause an IndexError
        prev_id = sorted_ids[0]
        for id in sorted_ids[1:]:
            if abs(id - prev_id) > 1:
                return prev_id + 1
        # if the end of the list has been reached without finding a free id, return the max id + 1
        return sorted_ids[-1] + 1
    except IndexError:
        # it means the list of ids is empty, so return simply 1
        return 1


if __name__ == '__main__':
    scenario = read_scenario(path="../task1/scenarios/rimea1.scenario")
    add_pedestrian(scenario)
