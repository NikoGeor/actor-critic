import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from functools import partial
from tqdm import tqdm

# Import the network architectures from the adjacent networks.py file
from .networks import ActorNetwork, CriticNetwork


@jax.jit
def exploration_policy(v, k, N, A_size):
    """Calculates the mixed exploration policy natively in JAX."""
    eta = 1.0 / (1.0 + jnp.square(jnp.log(k / N + 1.0)))
    return eta / A_size + (1.0 - eta) * v


@partial(jax.jit, static_argnames=('N', 'A_size'))
def _train_step(actor_ts, critic_ts, act_s, act_a, crit_s, crit_a, k, rng, p, r, gamma, xi, N, A_size):
    """The core XLA-compiled training step."""
    rng, r1, r2, r3, r4 = jax.random.split(rng, 5)

    # =========================
    # 1. Sample next states
    # =========================
    p_actor = p[act_s, act_a]
    p_critic = p[crit_s, crit_a]
    act_s_next = jax.random.categorical(r1, jnp.log(p_actor))
    crit_s_next = jax.random.categorical(r2, jnp.log(p_critic))

    # =========================
    # 2. Sample next actions
    # =========================
    actor_logits_next = jnp.squeeze(actor_ts.apply_fn({'params': actor_ts.params}, xi[act_s_next]), axis=-1)
    actor_pi_next = exploration_policy(jax.nn.softmax(actor_logits_next), k, N, A_size)
    act_a_next = jax.random.categorical(r3, jnp.log(actor_pi_next))

    critic_logits_next = jnp.squeeze(actor_ts.apply_fn({'params': actor_ts.params}, xi[crit_s_next]), axis=-1)
    critic_pi_next = exploration_policy(jax.nn.softmax(critic_logits_next), k, N, A_size)
    crit_a_next = jax.random.categorical(r4, jnp.log(critic_pi_next))

    # =========================
    # 3. Critic update (TD)
    # =========================
    def critic_loss_fn(params):
        q_sa = jnp.squeeze(critic_ts.apply_fn({'params': params}, xi[crit_s]))[crit_a]
        q_next = jnp.squeeze(critic_ts.apply_fn({'params': params}, xi[crit_s_next]))[crit_a_next]
        td_target = r[crit_s, crit_a] + gamma * jax.lax.stop_gradient(q_next)
        return jnp.square(td_target - q_sa)

    critic_grads = jax.grad(critic_loss_fn)(critic_ts.params)
    critic_ts = critic_ts.apply_gradients(grads=critic_grads)

    # =========================
    # 4. Actor update
    # =========================
    decay_factor = 1.0 / (1.0 + k / N)
    
    def actor_loss_fn(params):
        logits = jnp.squeeze(actor_ts.apply_fn({'params': params}, xi[act_s]), axis=-1)
        pi_soft = jax.nn.softmax(logits)
        log_pi_a = jnp.log(pi_soft[act_a] + 1e-10)
        
        q_a = jax.lax.stop_gradient(
            jnp.squeeze(critic_ts.apply_fn({'params': critic_ts.params}, xi[act_s]))[act_a]
        )
        return -log_pi_a * q_a * decay_factor

    actor_grads = jax.grad(actor_loss_fn)(actor_ts.params)
    actor_ts = actor_ts.apply_gradients(grads=actor_grads)

    return act_s_next, act_a_next, crit_s_next, crit_a_next, actor_ts, critic_ts, rng


class ActorCriticAlgorithm:

    def __init__(self, N, T, beta, alpha, zeta, mdp, seed=42):
        self.N = N
        self.T = T
        
        # Keep environment constants as JAX arrays
        self.p = jnp.asarray(mdp.p, dtype=jnp.float32)
        self.r = jnp.asarray(mdp.r, dtype=jnp.float32)
        self.rho0 = jnp.asarray(mdp.rho0, dtype=jnp.float32)
        self.gamma = jnp.asarray(mdp.gamma, dtype=jnp.float32)
        self.A_size = mdp.A_size

        # Create state-action pairs
        rep1 = np.repeat(mdp.X, mdp.A_size, axis=0)
        rep2 = np.repeat(mdp.A, mdp.X_size, axis=1).reshape(-1, mdp.da, order='F')
        XI = np.concatenate([rep1, rep2], axis=-1)
        XI = XI.reshape([mdp.X_size, mdp.A_size, -1])
        self.xi = jnp.asarray(XI, dtype=jnp.float32)

        # Initialize network architectures using Flax
        actor_model = ActorNetwork(N=N, beta=beta, outer_mean=0.0, inner_mean=0.0, outer_sd=1.0, inner_sd=1.0)
        critic_model = CriticNetwork(N=N, beta=beta, outer_mean=0.0, inner_mean=0.0, outer_sd=1.0, inner_sd=1.0)

        # Setup initial network weights and optimizers
        self.rng = jax.random.PRNGKey(seed)
        self.rng, act_key, crit_key = jax.random.split(self.rng, 3)
        
        dummy_xi = self.xi[0] 
        
        self.actor_ts = train_state.TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_model.init(act_key, dummy_xi)['params'],
            tx=optax.sgd(learning_rate=zeta * N**(2*beta - 2))
        )
        
        self.critic_ts = train_state.TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_model.init(crit_key, dummy_xi)['params'],
            tx=optax.sgd(learning_rate=alpha * N**(2*beta - 2))
        )

        self.actor_losses, self.critic_losses, self.bellman_losses = [], [], []
        self.Q, self.pi, self.rewards, self.critic_dists = [], [], [], []

    def initialize_chains(self):
        self.rng, r1, r2, r3, r4 = jax.random.split(self.rng, 5)

        act_s = jax.random.categorical(r1, jnp.log(self.rho0))
        crit_s = jax.random.categorical(r2, jnp.log(self.rho0))

        logits_act = jnp.squeeze(self.actor_ts.apply_fn({'params': self.actor_ts.params}, self.xi[act_s]), axis=-1)
        pi_act = exploration_policy(jax.nn.softmax(logits_act), 0.0, self.N, self.A_size)
        act_a = jax.random.categorical(r3, jnp.log(pi_act))

        logits_crit = jnp.squeeze(self.actor_ts.apply_fn({'params': self.actor_ts.params}, self.xi[crit_s]), axis=-1)
        pi_crit = exploration_policy(jax.nn.softmax(logits_crit), 0.0, self.N, self.A_size)
        crit_a = jax.random.categorical(r4, jnp.log(pi_crit))

        return act_s, act_a, crit_s, crit_a

    def losses(self, optimal_policy, mdp):
        logits = self.actor_ts.apply_fn({'params': self.actor_ts.params}, self.xi)
        ac = jax.nn.softmax(jnp.squeeze(logits, axis=-1))
        
        crit = jnp.squeeze(self.critic_ts.apply_fn({'params': self.critic_ts.params}, self.xi))

        q_fun = mdp.compute_q_function(ac)
        
        critic_loss = jnp.mean(jnp.square(q_fun - crit))
        actor_loss = jnp.mean(jnp.square(ac - optimal_policy))

        target = self.r + self.gamma * jnp.einsum('sb, sb, xas -> xa', crit, ac, self.p)
        bellman_loss = jnp.mean(jnp.square(crit - target))
        
        reward = q_fun[0, 1]
        
        q_function_truth = mdp.compute_q_function(optimal_policy)
        critic_dist = jnp.mean(jnp.square(q_function_truth - crit))

        return (np.asarray(critic_loss), np.asarray(actor_loss), np.asarray(bellman_loss), 
                np.asarray(ac), np.asarray(crit), np.asarray(reward), np.asarray(critic_dist))

    def fit(self, optimal_policy, mdp):
        act_s, act_a, crit_s, crit_a = self.initialize_chains()
        opt_pol = jnp.asarray(optimal_policy)

        for k in tqdm(range(self.N * self.T)):
            act_s, act_a, crit_s, crit_a, self.actor_ts, self.critic_ts, self.rng = _train_step(
                self.actor_ts, self.critic_ts, act_s, act_a, crit_s, crit_a, 
                float(k), self.rng, self.p, self.r, self.gamma, self.xi, self.N, self.A_size
            )
            
            if k % (self.N * self.T // 100) == 0:
                c, a, b, ac, cr, r, crd = self.losses(opt_pol, mdp)
                self.actor_losses.append(a)
                self.critic_losses.append(c)
                self.bellman_losses.append(b)
                self.Q.append(cr)
                self.pi.append(ac)
                self.rewards.append(r)
                self.critic_dists.append(crd)