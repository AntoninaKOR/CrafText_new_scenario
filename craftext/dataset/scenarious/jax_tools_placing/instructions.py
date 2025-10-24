from craftext.environment.scenarious.checkers.target_state import TargetState, ToolsPlacingState
from craftext.environment.craftext_constants import BlockType, MediumInventoryItems, Scenarios

def create_target_state(object_should_be):
    target_achievements = ToolsPlacingState(block_type=object_should_be)
    return TargetState(tools_placing=target_achievements)

easy = {
    "INSTRUCTION_PLACE_IRON_FURNACE_4_3": {
        "instruction": "Place Crafting Table and Furnace near Lava",
        "scenario_checker": Scenarios.TOOLS_PLACE,
        "instruction_paraphrases": [
            'Set up the Crafting Table and Furnace beside Lava',
            'Position a Crafting Table and Furnace close to Lava',
            'Put the Crafting Table and Furnace next to Lava',
            'Arrange the Crafting Table and Furnace near Lava',
            'Place both the Crafting Table and Furnace adjacent to Lava'
        ],
        "arguments": create_target_state(BlockType.LAVA),
        "str_check_lambda": "place_table_furnace(gd, ix)"
    }
}