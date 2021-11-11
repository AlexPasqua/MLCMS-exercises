import json
import warnings


def read_scenario(path="scenarios/rimea6.scenario"):
    """
    Loads a scenario file with json
    :param path: the path of the scenario file
    :return: the dictionary containing the scenario's data
    """
    with open(path, 'r') as f:
        scenario = json.load(f)
        return scenario


def add_pedestrian(scenario=None, scenario_path=None, output_path=None, id=None, find_min_free_id=True, targetIds=[],
                   radius=0.2, densityDependentSpeed=False, speedDistributionMean=1.34,
                   speedDistributionStandardDeviation=0.26, minimumSpeed=0.5, maximumSpeed=2.2, acceleration=2.0,
                   footstepHistorySize=4, searchRadius=1.0, walkingDirectionCalculation="BY_TARGET_CENTER",
                   walkingDirectionSameIfAngleLessOrEqual=45.0, nextTargetListIndex=0, position=(0, 0), velocity=(0, 0),
                   freeFlowSpeed=1.8522156059160915, followers=[], idAsTarget=-1, infectionStatus="SUSCEPTIBLE",
                   lastInfectionStatusUpdateTime=-1.0, pathogenAbsorbedLoad=0.0, groupIds=[], groupSizes=[],
                   agentsInGroup=[], traj_footsteps=[]):
    # 'scenario' and 'scenario_path' cannot be both None
    if scenario_path is None and scenario is None:
        raise AttributeError("One of 'scenario' and 'scenario_path' must be not None, got both None")

    # if the scenario path is not passed, than an output path is mandatory (otherwise the scenario path is used as output path too)
    elif scenario_path is None and output_path is None:
        raise AttributeError("One of 'scenario_path' and 'output_path' must be not None, got both None")

    # if both the scenario and its path are passed, only the scenario is going to be used and the path is ignored
    elif scenario_path is not None and scenario is not None:
        msg = "Both the scenario and the path to its file were passed to the function 'add_pedestrian'. " \
              "Only the scenario is going to be used, it will not be read again from file"
        warnings.warn(msg, RuntimeWarning)

    # if scenario_path is not None, read the scenario from file
    if scenario_path is not None:
        scenario = read_scenario(scenario_path)

    # if the pedestrian's id is not passed, find a free one
    if id is None:
        id = find_free_id(scenario, find_min_free_id=True)

    # create a dictionary with the pedestrian's data (in the format used in the scenario's json file)
    ped = {
        "attributes": {
            "id": id,
            "radius": radius,
            "densityDependentSpeed": densityDependentSpeed,
            "speedDistributionMean": speedDistributionMean,
            "speedDistributionStandardDeviation": speedDistributionStandardDeviation,
            "minimumSpeed": minimumSpeed,
            "maximumSpeed": maximumSpeed,
            "acceleration": acceleration,
            "footstepHistorySize": footstepHistorySize,
            "searchRadius": searchRadius,
            "walkingDirectionCalculation": walkingDirectionCalculation,
            "walkingDirectionSameIfAngleLessOrEqual": walkingDirectionSameIfAngleLessOrEqual
        },
        "source": None,
        "targetIds": targetIds,
        "nextTargetListIndex": nextTargetListIndex,
        "isCurrentTargetAnAgent": False,
        "position": {
            "x": position[0],
            "y": position[1]
        },
        "velocity": {
            "x": velocity[0],
            "y": velocity[1]
        },
        "freeFlowSpeed": freeFlowSpeed,
        "followers": followers,
        "idAsTarget": idAsTarget,
        "isChild": False,
        "isLikelyInjured": False,
        "psychologyStatus": {
            "mostImportantStimulus": None,
            "threatMemory": {
                "allThreats": [],
                "latestThreatUnhandled": False
            },
            "selfCategory": "TARGET_ORIENTED",
            "groupMembership": "OUT_GROUP",
            "knowledgeBase": {
                "knowledge": [],
                "informationState": "NO_INFORMATION"
            },
            "perceivedStimuli": [],
            "nextPerceivedStimuli": []
        },
        "healthStatus": {
            "infectionStatus": infectionStatus,
            "lastInfectionStatusUpdateTime": lastInfectionStatusUpdateTime,
            "pathogenAbsorbedLoad": pathogenAbsorbedLoad,
            "startBreatheOutPosition": None,
            "respiratoryTimeOffset": -1.0,
            "breathingIn": False,
            "pathogenEmissionCapacity": -1.0,
            "pathogenAbsorptionRate": -1.0,
            "minInfectiousDose": -1.0,
            "exposedPeriod": -1.0,
            "infectiousPeriod": -1.0,
            "recoveredPeriod": -1.0
        },
        "groupIds": groupIds,
        "groupSizes": groupSizes,
        "agentsInGroup": agentsInGroup,
        "trajectory": {"footSteps": traj_footsteps},
        "modelPedestrianMap": None,
        "type": "PEDESTRIAN"
    }

    # "scenario['scenario']['topography']['dynamicElements']" gives the list of the pedestrians in the scenario;
    # append to it the new pedestrian just created
    scenario['scenario']['topography']['dynamicElements'].append(ped)

    if output_path is None:  # if output_path is None, use scenario_path
        output_path = scenario_path
    elif not output_path.endswith(".scenario"):  # add ".scenario" suffix to the output path if not present
        output_path += ".scenario"

    # write the scenario file with the new pedestrian
    with open(output_path, 'w') as f:
        json.dump(scenario, f, indent='  ')


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
        return max(busy_ids) + 1  # simply return the max busy id + 1 (which will be free)

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
    add_pedestrian(
        scenario_path="../task1/scenarios/rimea1.scenario",
        output_path="scenarios/pippo.scenario",
        position=(40, 40)
    )
