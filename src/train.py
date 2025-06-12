# train_flax.py
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import tensorflow_datasets as tfds
import time
import numpy as np

from src.model import MLP, SSM
import src.data_load as dat
import src.utils.ml_functions as ml
import src.utils.helper_functions as hf
import src.utils.visualize as viz


def train(
        model_type="MLP", dataset="mnist", key_seed=0, epochs=5, batch_size=128, hidden_dim=64,
        truncate=None, lr = 1e-3, decay = None, decay_epoch_freq=10, hidden_layers=[128, 64, 10]
        ):

    X_train, y_train, y_train_as_idx, X_test, y_test, y_test_as_idx = dat.load_data(
        dataset='psMNIST', truncate=truncate)


    key = jax.random.PRNGKey(key_seed)
    if model_type=="MLP":
        # Define model and training state
        model = MLP(hidden_sizes=hidden_layers)
        variables = model.init(key, X_train[:1])['params']

    elif model_type=="SSM":
        model = SSM(hidden_dim=hidden_dim, input_dim=1, output_dim=10, seq_len=784)
        x = jnp.ones(X_train[:batch_size].reshape(batch_size, 784, 1).shape) 
        params = model.init(key, x)['params']

    if decay is not None:
        schedule = optax.exponential_decay(
            init_value=lr,
            transition_steps=decay_epoch_freq*(len(X_train) // batch_size),
            decay_rate=decay,
            staircase=True
        )
        optimizer = optax.adam(learning_rate=schedule)
    else:
        optimizer = optax.adam(lr)

    class State(TrainState):
        pass

    state = State.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    savename = f'{model_type}_{dataset}_depth_state{hidden_dim}_b{batch_size}_lr{lr}'
    print(savename)

    train_fn = ml.train_step

    accs = []
    training_loss = []
    hyper = None,None,None

    update_count = 0
    for epoch in range(epochs):
        epoch_loss = []
        t1 = time.perf_counter()
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            loss, state, hyper = train_fn(state, x_batch, y_batch, hyper)
            epoch_loss.append(loss)
            update_count+=1

        training_loss.append(np.mean(epoch_loss))
        acc = ml.compute_accuracy(state, X_test, y_test)
        accs.append(acc)
        t2 = time.perf_counter()
        print_dat = (epoch, training_loss[-1], acc, t2-t1)
        hf.print_progress(print_dat,precision=3)

    hf.picklit(
        [state.params,accs],
        'params',
        savename
        )
    
    performance = viz.performance_plot(accs,training_loss,show=False)
    hf.save_fig(performance,"figs",savename)
