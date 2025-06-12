import flax.linen as nn
import jax.numpy as jnp
from typing import Callable
from src.utils.ml_functions import make_discrete_HiPPO_nojit

class MLP(nn.Module):
    hidden_sizes: list[int]

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_sizes[:-1]:
            x = nn.relu(nn.Dense(size)(x))
        x = nn.Dense(self.hidden_sizes[-1])(x)
        return x


class CustomModel():
    pass


class SSMSingleCell(nn.Module):
    hidden_dim: int
    input_dim: int
    transition_fn: Callable = nn.tanh  # nonlinearity for state update

    @nn.compact
    def __call__(self, carry, x):
        C = self.param("C", nn.initializers.xavier_uniform(), (self.input_dim, self.hidden_dim))
        A, B = make_discrete_HiPPO_nojit(self.hidden_dim,784)
        h = carry
        u = x.reshape(h.shape[0],) 
        # h_next = self.transition_fn(jnp.dot(A, h) + jnp.dot(B, u))  # state update
        h_next = self.transition_fn(jnp.dot(A, h.T).T + (B * u).T)
        y = jnp.dot(C, h_next.T).T  # output
        return h_next, y


class SSM(nn.Module):
    hidden_dim: int
    input_dim: int
    output_dim: int
    seq_len: int

    @nn.compact
    def __call__(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size = x.shape[0]
        initial_state = jnp.zeros((batch_size, self.hidden_dim))

        # Use `flax.linen.scan` to unroll over time
        ssm_cell = nn.scan(
            SSMSingleCell,
            variable_broadcast="params",  # share weights across time
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
            # name="ssm_scan"
        )(self.hidden_dim, self.input_dim)(initial_state, x)

        final_state, y_seq = ssm_cell  # y_seq: [batch, seq_len, input_dim]

        # Optional: map to desired output dim
        y_proj = nn.Dense(self.output_dim)(y_seq)

        return y_proj[:,-1,:]
