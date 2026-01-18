"""Elementary Cellular Automata module.

This module implements Elementary Cellular Automata, one-dimensional cellular automata where each
cell's next state depends on its current state and that of its two immediate neighbors. Elementary
Cellular Automata are classified by Wolfram rule numbers (0-255), which define the transition
function for all possible three-cell neighborhoods.
"""

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem, Input, State
from cax.utils import clip_and_uint8

from ohm_perceive import ohm_perceive
from ohm_update import ohm_update


class ohm_basic(ComplexSystem):

	def __init__(
		self,
		*,
		rule_number: int = 110,
		rngs: nnx.Rngs		
	):
		self.perceive = ohm_perceive(rngs=rngs)
		self.update = ohm_update(rule_number=rule_number)

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render state to RGB image.

		Converts the one-dimensional cellular automaton state to an RGB visualization
		by replicating the single-channel state values across all three color channels,
		resulting in a grayscale image.

		Args:
			state: Array with shape (num_steps, width, 1) representing the
				cellular automaton state, where each cell contains a value in [0, 1].

		Returns:
			RGB image with dtype uint8 and shape (num_steps, width, 3), where cell values are mapped
				to grayscale colors in the range [0, 255].

		"""
		rgb = jnp.repeat(state, 3, axis=-1)

		return clip_and_uint8(rgb)
