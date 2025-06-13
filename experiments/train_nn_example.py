from src.train import train

def main():
    train(
        model_type="MLP",
        dataset="MNIST",
        epochs=100,
        batch_size=128,
        hidden_dim=64,
        truncate=None,
        )
    # train(
    #     model_type="SSM",
    #     dataset="psMNIST",
    #     epochs=100,
    #     batch_size=128,
    #     hidden_dim=64,
    #     truncate=0.1
    #     )
if __name__=="__main__":
    main()