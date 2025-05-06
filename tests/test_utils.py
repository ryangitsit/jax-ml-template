import jax
import jax.numpy as jnp

def create_one_hot_labels(batch_size, num_classes, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    class_indices = jax.random.randint(key, (batch_size,), 0, num_classes)
    return jax.nn.one_hot(class_indices, num_classes)