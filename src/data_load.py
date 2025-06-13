import tensorflow_datasets as tfds
import jax.numpy as jnp
import numpy as np
import warnings
import jax

def permute_mnist(flat_images, seed=42):
    """
    Convert MNIST images to Permuted Sequential MNIST format.

    Args:
        images: numpy array of shape (num_samples, 28, 28)
        seed: random seed to fix the permutation

    Returns:
        permuted_images: numpy array of shape (num_samples, 784)
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(784)

    permuted_images = flat_images[:, permutation]  # Apply same perm to all
    return permuted_images

def load_data(dataset='mnist', truncate=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if dataset == 'MNIST' or dataset == 'mnist':
        # Load MNIST
        ds = tfds.load('mnist', split='train', batch_size=-1)
        test_ds = tfds.load('mnist', split='test', batch_size=-1)
        train = tfds.as_numpy(ds)
        test = tfds.as_numpy(test_ds)

        x_train, y_train_as_idx = train['image'], train['label']
        x_test, y_test_as_idx   = test['image'], test['label']

        # Preprocess
        X_train = x_train.reshape(-1, 28*28) / 255.0
        X_test  = x_test.reshape(-1, 28*28) / 255.0
        y_train = jax.nn.one_hot(y_train_as_idx, 10)
        y_test  = jax.nn.one_hot(y_test_as_idx, 10)

    elif dataset=="psMNIST":
        # X_train = permute_mnist(X_train)
        # X_test  = permute_mnist(X_test)

        X_train, y_train, y_train_as_idx = prepare_psmnist_data(dataset='train',seq_len=784)
        X_test, y_test, y_test_as_idx = prepare_psmnist_data(dataset='test',seq_len=784)

    else:
        raise ValueError(f"{dataset} not a supported dataset.")
    
    if truncate is not None:
        train_trunc = int(X_train.shape[0]*truncate)
        test_trunc = int(X_test.shape[0]*truncate)

        X_train = X_train[:train_trunc]
        y_train = y_train[:train_trunc]
        y_train_as_idx = y_train_as_idx[:train_trunc]
        X_test = X_test[:test_trunc]
        y_test = y_test[:test_trunc]
        y_test_as_idx = y_test_as_idx[:test_trunc]


    return X_train, y_train, y_train_as_idx, X_test, y_test, y_test_as_idx


def prepare_psmnist_data(dataset='train',seq_len=784):
    ds = tfds.load('mnist', split=dataset, as_supervised=True)
    images = []
    labels = []

    for img, label in tfds.as_numpy(ds):
        img = img.astype(np.float32).reshape(-1) / 255.0  # flatten and normalize
        images.append(img)
        labels.append(label)

    images = np.stack(images)  # (60000, 784)
    labels = np.array(labels)  # (60000,)

    # Permute the sequence (same permutation for all samples)
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(seq_len)
    images = images[:, perm]

    # Add input_dim=1
    images = images[..., None]  # (60000, 784, 1)

    return images, jax.nn.one_hot(labels, 10), labels