import jax
from jax import (
    numpy as jnp,
    lax
)

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state import LocalizaPlacingState
from functools import partial



def checker_water(game_data: Union[GameDataClassic, GameData], target_state: LocalizaPlacingState) -> jax.Array:

    return found_all(game_data=game_data)


from functools import partial



@jax.jit
def found_all(
    game_data: Union[GameDataClassic, GameData], 
    object_to_find=3,    
) -> jax.Array:
    return jnp.all(jnp.array(game_data.is_visited_water))