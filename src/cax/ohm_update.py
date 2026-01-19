"""Elementary Cellular Automata update module.

This module implements the update rule for Elementary Cellular Automata based on Wolfram codes.
Each cell's next state is determined by looking up its three-cell neighborhood configuration
in the Wolfram code lookup table.

"""

import jax
import jax.numpy as jnp
from jax import Array

from cax.core import Input, State
from cax.core.perceive import Perception
from cax.core.update import Update


class ohm_update(Update):
	"""Elementary Cellular Automata update rule.

	Applies the Wolfram rule by matching each cell's three-cell neighborhood against all
	possible configurations and selecting the corresponding output value from the Wolfram code.

	"""

	def __init__(self, rule_number: int = 110):
		self.configurations = jnp.array(
			[
				[1, 1, 1],
				[1, 1, 0],
				[1, 0, 1],
				[1, 0, 0],
				[0, 1, 1],
				[0, 1, 0],
				[0, 0, 1],
				[0, 0, 0],
			],
			dtype=jnp.float32,
		)
		self.wolfram_code = ((rule_number >> 7 - jnp.arange(8)) & 1).astype(jnp.float32)		 

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Process the current state, perception, and input to produce a new state.
		Args:
			state: Current state (unused, next state computed solely from perception).
			perception: Array with shape (..., width, 3) containing the three-cell neighborhood
				(left, self, right) for each cell.
			input: Optional input (unused in this implementation).

		Returns:
			Next state with shape (..., width, 1) containing the updated cell values.
		"""
		return jnp.vectorize(self.update_cell, signature="(3)->(1)")(perception)


	def update_cell(self, neighborhood: Array) -> Array:
		"""Determines the next state for a single cell based on its neighborhood."""
		matches = jnp.all(neighborhood == self.configurations, axis=-1)
		return jnp.sum(matches * self.wolfram_code, keepdims=True)
