import jax.numpy as jnp
import jax

import src.utils.ml_functions as ml
import tests.test_utils as tu

def test_loss_CE():
    batch_size = 32
    num_classes = 10

    # test random output against random labels
    rng = jax.random.PRNGKey(1)
    test_logits = jax.random.uniform(rng, (batch_size, num_classes))
    test_labels = tu.create_one_hot_labels(batch_size, num_classes, key=None)
    test_loss = ml.loss_CE(test_logits, test_labels)
    assert test_loss.shape == ()
    assert test_loss > 0
    assert test_loss == ml.loss_CE_manual(test_logits,test_labels)
    
    test_loss_correct   = ml.loss_CE(test_labels, test_labels)
    assert test_loss_correct < test_loss