from typing import List, Tuple, Optional, Union
from flax import struct
from jax import (
    numpy as jnp,
    lax
)
from functools import partial
import jax
from scipy import ndimage as ndi
import numpy as np
#from craftext_constants import BlockType

MAX_RADIUS = 5
REGION_SIZE = 2 * MAX_RADIUS + 1

def masked_region_map(game_map, x, y, fill_value=-1):
    H, W = game_map.shape

    # coordinate grids
    yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')

    # distance (square or circle)
    dist = jnp.sqrt((xx - x)**2 + (yy - y)**2)

    # mask inside radius
    mask = dist <= MAX_RADIUS

    # apply mask
    masked = jnp.where(mask, game_map, fill_value)
    return masked

@struct.dataclass
class PlayerVariables:
    player_position: jax.Array
    player_direction: int
    player_health: int
    player_food: int
    player_drink: int
    player_energy: int
    is_sleeping: bool
    player_recover: float
    player_hunger: float
    player_thirst: float
    player_fatigue: float
    light_level: float
    state_rng: jax.Array
    timestep: int
    
@struct.dataclass
class PlayerAchievements:
    achievements: List[str]

@struct.dataclass
class PlayerInventory:
    inventory = 1
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    sapling: int
    wood_pickaxe: int
    stone_pickaxe: int
    iron_pickaxe: int
    wood_sword: int
    stone_sword: int
    iron_sword: int
     # Just for jax for correct invemtory check
    pickaxe: jax.Array
    sword: jax.Array
    bow: jax.Array
    arrows: jax.Array
    armour: jax.Array
    torches: jax.Array
    ruby: jax.Array
    sapphire: jax.Array
    potions: jax.Array
    books: jax.Array



@struct.dataclass
class GameMap:
    game_map: jax.Array


@struct.dataclass
class PlayerState:
    variables: PlayerVariables
    achievements: PlayerAchievements
    inventory: PlayerInventory
    action: int
    map: GameMap

    @classmethod
    def from_state(cls, state, action):
        variables = PlayerVariables(
            player_position=jnp.array(state.player_position) if hasattr(state, 'player_position') else None,
            player_direction=state.player_direction,
            player_health=state.player_health,
            player_food=state.player_food,
            player_drink=state.player_drink,
            player_energy=state.player_energy,
            is_sleeping=state.is_sleeping,
            player_recover=state.player_recover,
            player_hunger=state.player_hunger,
            player_thirst=state.player_thirst,
            player_fatigue=state.player_fatigue,
            light_level=state.light_level,
            state_rng=state.state_rng,
            timestep=state.timestep,
        )

        achievements = PlayerAchievements(
            achievements=jnp.array(state.achievements) if hasattr(state, 'achievements') else None
        )

        inventory = PlayerInventory(
            wood=state.inventory.wood,
            stone=state.inventory.stone,
            coal=state.inventory.coal,
            iron=state.inventory.iron,
            diamond=state.inventory.diamond,
            sapling=state.inventory.sapling,
            wood_pickaxe=state.inventory.wood_pickaxe,
            stone_pickaxe=state.inventory.stone_pickaxe,
            iron_pickaxe=state.inventory.iron_pickaxe,
            wood_sword=state.inventory.wood_sword,
            stone_sword=state.inventory.stone_sword,
            iron_sword=state.inventory.iron_sword,
            # Just for jax for correct invemtory check
            pickaxe=state.inventory.iron,
            sword=state.inventory.iron,
            bow=state.inventory.iron,
            arrows=state.inventory.iron,
            armour=state.inventory.iron,
            torches=state.inventory.iron,
            ruby=state.inventory.iron,
            sapphire=state.inventory.iron,
            potions=state.inventory.iron,
            books=state.inventory.iron,
        )
        game_map = GameMap(
            game_map=jnp.array(state.map) if hasattr(state, 'map') else None
        )

        return cls(
            variables=variables,
            achievements=achievements,
            inventory=inventory,
            map=game_map,
            action=action
        )

# def find_water(map, connectivity=8):
#     """
#     finding connectivity components of mask using 8 or 4 связность
#     """
#     mask = (map == 3)

#     if connectivity == 8:
#         structure = jnp.array([[1,1,1],
#                             [1,1,1],
#                             [1,1,1]], dtype=bool)
#     else:
#         structure = jnp.array([[0,1,0],
#                             [1,1,1],
#                             [0,1,0]], dtype=bool)

#     water_masks, n = ndi.label(np.asarray(mask), structure=structure)
#     return jnp.array(water_masks)

def find_water(map, connectivity=8, max_iter=4096):
    mask = (map == 3)
    structure = jnp.ones((3,3), bool) if connectivity==8 else jnp.array([[0,1,0],[1,1,1],[0,1,0]], bool)

    labels = jnp.zeros_like(mask, dtype=jnp.int32)
    current_label = 0

    def cond_fn(carry):
        mask, labels, current_label = carry
        unlabeled = mask & (labels == 0)
        return unlabeled.any() & (current_label < max_iter)

    def body_fn(carry):
        mask, labels, current_label = carry
        unlabeled = mask & (labels == 0)
        seed_idx = jnp.argmax(unlabeled)
        H, W = mask.shape
        y = seed_idx // W
        x = seed_idx % W
        new_label = current_label + 1

        region = jnp.zeros_like(mask, bool)
        frontier = jnp.zeros_like(mask, bool).at[y, x].set(True)

        def fill_cond(state):
            _, f = state
            return f.any()

        def fill_body(state):
            region, f = state
            dilated = lax.conv_general_dilated(
                f.astype(jnp.int32)[None,None,:,:],
                structure.astype(jnp.int32)[None,None,:,:],
                (1,1),
                "SAME"
            )[0,0] > 0
            new_frontier = dilated & mask & (labels == 0) & (~region)
            region = region | new_frontier
            return (region, new_frontier)

        region, _ = lax.while_loop(fill_cond, fill_body, (region, frontier))
        labels = jnp.where(region, new_label, labels)
        return (mask, labels, new_label)

    mask, labels, n = lax.while_loop(cond_fn, body_fn, (mask, labels, current_label))
    return labels, n


def update_is_visited_water(state, water_sources, n):
    x, y = state.variables.player_position
    print(state.map.game_map.shape)
    MAX_WATER_LABELS = 64 
    region = masked_region_map(state.map.game_map, x, y)
    visible_water_mask = (region == 3)
    res_init = jnp.zeros((MAX_WATER_LABELS,), dtype=jnp.int32)

    def body_fn(i, res):
        # i is traced index (0..n-1)
        label = i + 1  # labels are 1..n
        visible = jnp.any((water_sources == label) & visible_water_mask)
        return res.at[i].set(visible.astype(jnp.int32))

    # Use lax.fori_loop to iterate from 0..n-1
    res = lax.fori_loop(0, n, body_fn, res_init)
    return jnp.any(res)
            

@struct.dataclass
class GameDataClassic:
    states: list
    water_sources: list
    is_visited_water: list

    @classmethod
    def from_state(cls, previos_state, current_state, action):
        player_state_current = PlayerState.from_state(current_state, action)
        player_state_previos = PlayerState.from_state(previos_state, action)
        water_sources, n = find_water(current_state.map) 
        is_visited_water=update_is_visited_water(player_state_current, water_sources, n)
        return cls(states=[player_state_current, player_state_previos],
                   water_sources=water_sources,
                   is_visited_water=is_visited_water)
    