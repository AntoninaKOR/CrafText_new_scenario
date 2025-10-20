from craftext.environment.scenarious.checkers.target_state import TargetState, ExploringWaterState
from craftext.environment.craftext_constants import Scenarios, BlockType

def create_target_state(object_to_find):
    target_achievements = ExploringWaterState(object_to_find)
    return TargetState(exploring_water=target_achievements)

easy = {
    "INSTRUCTION_FIND_ALL_WATER_SOURCES": {
        "instruction": "Explore the world and find all water sources",
        "scenario_checker": Scenarios.EXPLORE_WATER,
        "instruction_paraphrases": [
            "Take a walk around and see if you can find any rivers, lakes, or ponds.",
            "Look around the area and investigate where water might be flowing or collected.",
            "Wander through the environment and discover where water sources are located.",
            "Move freely through the world and check each corner for signs of water.",
            "Travel around the map and visit different spots to look for streams or pools.",
            "Roam the landscape and search for places where water gathers — valleys and lowlands are good clues.",
            "Survey the environment carefully.",
            "Navigate through the terrain and see if you can spot any bodies of water.",
            "Go out and examine the world, keeping your eyes open for water.",
            "Step outside and explore nearby areas; you might find hidden wells or ponds.",
            "Stroll through the world and look for rivers, streams, or any flowing water.",
            "Leave your starting point and venture out — water is often found beyond familiar ground.",
            "Uncover hidden parts of the map by walking around and checking for pools or springs.",
            "Explore different directions and note any spots that show signs of water.",
            "Walk through the terrain and learn where water sources exist across the land.",
            "Discover your surroundings by moving through them and identifying every place that holds water."
            ],
        "arguments": create_target_state(BlockType.WATER)
    }
}
