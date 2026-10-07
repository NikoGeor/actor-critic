import jax
import jax.numpy as jnp
import flax.linen as nn

class NNLayer(nn.Module):
    # Model hyperparameters
    N: int
    beta: float
    outer_mean: float
    outer_sd: float
    inner_mean: float
    inner_sd: float

    @nn.compact
    def __call__(self, inputs, training=False):
        # Custom initializers to apply a specific standard deviation and mean shift
        def inner_init(key, shape, dtype=jnp.float32):
            base_init = jax.nn.initializers.truncated_normal(stddev=self.inner_sd)
            return base_init(key, shape, dtype) + self.inner_mean

        def outer_init(key, shape, dtype=jnp.float32):
            base_init = jax.nn.initializers.truncated_normal(stddev=self.outer_sd)
            return base_init(key, shape, dtype) + self.outer_mean

        # Initialize and retrieve parameters inline 
        # (Shapes are dynamically inferred from the inputs on the first pass)
        W = self.param('inner_layer', inner_init, (inputs.shape[-1], self.N))
        c = self.param('outer_layer', outer_init, (self.N, 1))

        # Forward pass operations
        inner = jnp.matmul(inputs, W)
        act = jax.nn.sigmoid(inner)
        outer = jnp.matmul(act, c)

        out = outer / (self.N ** self.beta)
        
        return out

class ActorNetwork(nn.Module):
    # Network hyperparameters
    N: int
    beta: float
    outer_mean: float
    inner_mean: float
    outer_sd: float
    inner_sd: float

    def setup(self):
        # Sub-modules instantiated in setup() are automatically tracked 
        # and nested into the network's overarching parameter dictionary.
        self.layer1 = NNLayer(
            N=self.N,
            beta=self.beta,
            outer_mean=self.outer_mean,
            outer_sd=self.outer_sd,
            inner_mean=self.inner_mean,
            inner_sd=self.inner_sd
        )

    def __call__(self, input_tensor, training=False):
        x = self.layer1(input_tensor, training=training)
        return x

class CriticNetwork(nn.Module):
    # Network hyperparameters 
    N: int
    beta: float
    outer_mean: float
    inner_mean: float
    outer_sd: float
    inner_sd: float

    def setup(self):
        self.layer1 = NNLayer(
            N=self.N,
            beta=self.beta,
            outer_mean=self.outer_mean,
            outer_sd=self.outer_sd,
            inner_mean=self.inner_mean,
            inner_sd=self.inner_sd
        )

    def __call__(self, input_tensor, training=False):
        x = self.layer1(input_tensor, training=training)
        return x