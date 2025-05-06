import jax.numpy as jnp
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