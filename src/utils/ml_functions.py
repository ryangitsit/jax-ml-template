import jax.numpy as jnp
from jax.numpy.linalg import inv
import jax
import optax


@jax.jit
def innerA(X, y):
    return jnp.dot(X, y)

@jax.jit
def innerB(X, y):
    return jnp.sum(X * y)

def make_outer_fn(inner_fn):
    @jax.jit
    def outer_fn(X, y):
        return inner_fn(X, y) * 2
    return outer_fn

@jax.jit
def loss_MSE(logits, labels):
    return optax.losses.l2_loss(logits,labels).mean()

@jax.jit
def loss_CE_manual(logits, labels):
    log_softmax_probs = jax.nn.log_softmax(logits, axis=-1)
    per_example_loss = -jnp.sum(labels * log_softmax_probs, axis=-1) 
    return jnp.mean(per_example_loss)

@jax.jit
def loss_CE(logits, labels):
    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

@jax.jit
def create_one_hot_labels(num_classes, labels_by_index):
    return jax.nn.one_hot(labels_by_index, num_classes)

@jax.jit
def train_step(state, x, y, hyper):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = optax.softmax_cross_entropy(logits, y).mean()
        return loss
    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    loss, grads = loss_and_grad_fn(state.params)
    return loss, state.apply_gradients(grads=grads), hyper

@jax.jit
def compute_accuracy(state, x, y):
    logits = state.apply_fn({'params': state.params}, x)
    predictions = jnp.argmax(logits, axis=-1)
    targets = jnp.argmax(y, axis=-1)
    return jnp.mean(predictions == targets)


def discretize_nojit(A, B, step):
    I  = jnp.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb

def make_HiPPO_nojit(N):
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    B = B.reshape(len(B),1)
    return -A, B

def make_discrete_HiPPO_nojit(N,L):
    A, B = make_HiPPO_nojit(N)
    return discretize_nojit(A, B, step=1.0 / L)