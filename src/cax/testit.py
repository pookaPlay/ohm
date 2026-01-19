import cv2
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ohm_basic import ohm_basic

seed = 0

num_steps = 128
spatial_dims = (64,)

rngs = nnx.Rngs(seed)

ohm = ohm_basic(rngs=rngs, rule_number=110)

def sample_state():
	"""Sample a state with a single active cell."""
	state = jnp.zeros((*spatial_dims, 1))
	return state.at[spatial_dims[0] // 2].set(1.0)

state_init = sample_state()
state_final = ohm(state_init, num_steps=num_steps, sow=True)

intermediates = nnx.pop(ohm, nnx.Intermediate)
states = intermediates.state[0]

states = jnp.concatenate([state_init[None], states])
frame = ohm.render(states)

cv2.imshow("OHM", cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
