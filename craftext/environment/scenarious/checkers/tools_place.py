import jax
from jax import (
    numpy as jnp,
    lax
)

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic
from craftext.environment.craftext_constants import BlockType
from craftext.environment.scenarious.checkers.target_state import ToolsPlacingState
from functools import partial


@partial(jax.jit, static_argnames=['max_radius'])
def safe_dynamic_slice(game_map, x, y, radius, max_radius):
    full_region_size = 2 * max_radius + 1

    x_padded = x + max_radius
    y_padded = y + max_radius

    region = lax.dynamic_slice(
        game_map,
        start_indices=(x_padded - max_radius, y_padded - max_radius),
        slice_sizes=(full_region_size, full_region_size)
    )

    coord_range = jnp.arange(full_region_size) - max_radius
    mask_x = jnp.abs(coord_range) <= radius
    mask_y = mask_x[:, None]
    mask = mask_x & mask_y

    region_masked = jnp.where(mask, region, -1)
    return region_masked


def checker_tools(game_data: Union[GameDataClassic, GameData], target_state: ToolsPlacingState) -> jax.Array:
    
    block_name = target_state.block_type
    return place_table_furnace(game_data=game_data, object_name=block_name)


from functools import partial

MAX_RADIUS = 5
REGION_SIZE = 2 * MAX_RADIUS + 1

@jax.jit
def place_table_furnace(
    game_data: Union[GameDataClassic, GameData], 
    object_name: str  
) -> jax.Array:
    # Extract a square region of size REGION_SIZE×REGION_SIZE around the player
    x, y = game_data.states[0].variables.player_position
    padded_map = jnp.pad(
        game_data.states[0].map.game_map,
        ((MAX_RADIUS, MAX_RADIUS), (MAX_RADIUS, MAX_RADIUS)),
        constant_values=-1  # any out-of-bounds marker
    )
    region = lax.dynamic_slice(
        padded_map,
        (x, y),
        (REGION_SIZE, REGION_SIZE)
    )  # shape [REGION_SIZE, REGION_SIZE]

    # Build boolean masks for the target and the object
    #    (you can compare against .value directly, or if you use integer codes, omit .value)
    ct_mask = (region == BlockType.CRAFTING_TABLE)
    f_mask = (region == BlockType.FURNACE)
    obj_mask = (region == object_name)

    # Check if there is any position (i, j) where both tgt_mask and shifted_obj are True
    hit = ct_mask & f_mask & obj_mask

    # If a `need_to_achieve` flag is required, handle it externally via select()
    return jnp.any(hit)