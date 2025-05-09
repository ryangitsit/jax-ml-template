# train_flax.py
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import tensorflow_datasets as tfds
import time

from src.model import MLP, SSM, MultiLayerSSM
import src.data_load as dat
import src.utils.ml_functions as ml
import src.utils.helper_functions as hf


def train(model_type="MLP", dataset="mnist", key_seed=0,epochs=5,batch_size=128,hidden_dim=64):

    X_train, y_train, y_train_as_idx, X_test, y_test, y_test_as_idx = dat.load_data(dataset='psMNIST')


    key = jax.random.PRNGKey(key_seed)
    if model_type=="MLP":
        # Define model and training state
        model = MLP(hidden_sizes=[128, 64, 10])
        variables = model.init(key, X_train[:1])['params']

    elif model_type=="SSM":
        model = SSM(hidden_dim=hidden_dim, input_dim=1, output_dim=10, seq_len=784)
        x = jnp.ones(X_train[:batch_size].reshape(batch_size, 784, 1).shape) 
        params = model.init(key, x)['params']

    elif model_type=='DeepSSM':
        model = MultiLayerSSM(input_dim=1, hidden_dim=hidden_dim, output_dim=10, num_layers=2)
        x = jnp.ones(X_train[:batch_size].reshape(batch_size, 784, 1).shape) 
        params = model.init(key, x)['params']

    optimizer = optax.adam(1e-3)

    class State(TrainState):
        pass

    state = State.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    accs = []
    for epoch in range(epochs):
        t1 = time.perf_counter()
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            state = ml.train_step(state, x_batch, y_batch)

        acc = ml.compute_accuracy(state, X_test, y_test)
        accs.append(acc)
        t2 = time.perf_counter()
        print(f"Epoch {epoch+1}, Test Accuracy: {acc:.4f}  - runtime {t2-t1:.3}")

    hf.picklit([state.params,accs],'params','multilayer_ssm_psmnist')

# train(model_type="SSM",dataset="psMNIST")