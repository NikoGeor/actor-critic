import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class MDP:
    def __init__(self, states, actions, init_dist, rewards, transitions, discount):
        # Convert all inputs to static JAX arrays upon initialization
        self.r = jnp.asarray(rewards, dtype=jnp.float32)
        self.rho0 = jnp.asarray(init_dist, dtype=jnp.float32)
        self.p = jnp.asarray(transitions, dtype=jnp.float32)
        self.X = jnp.asarray(states, dtype=jnp.float32)
        self.A = jnp.asarray(actions, dtype=jnp.float32)
        
        self.X_size = states.shape[0]
        self.A_size = actions.shape[0]
        self.dx = states.shape[1]
        self.da = actions.shape[1]
        self.gamma = discount

    def tree_flatten(self):
        # 'children' are the dynamic arrays JAX tracks
        children = (self.r, self.rho0, self.p, self.X, self.A)
        # 'aux_data' are the static metadata values
        aux_data = (self.X_size, self.A_size, self.dx, self.da, self.gamma)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstructs the object during XLA compilation
        obj = object.__new__(cls)
        obj.r, obj.rho0, obj.p, obj.X, obj.A = children
        obj.X_size, obj.A_size, obj.dx, obj.da, obj.gamma = aux_data
        return obj

    @jax.jit
    def compute_q_function(self, policy, n_iter=10000):
        # XLA-compiled fori_loop prevents python unrolling of the 10,000 steps
        def body_fn(i, Q):
            return self.r + self.gamma * jnp.einsum('xas, sb, sb -> xa', self.p, policy, Q)

        Q_init = jnp.zeros((self.X_size, self.A_size))
        Q_final = jax.lax.fori_loop(0, n_iter, body_fn, Q_init)
        
        return Q_final