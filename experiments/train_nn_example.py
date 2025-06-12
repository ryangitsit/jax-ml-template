from src.train import train

def main():
    train(
        model_type="SSM",
        dataset="psMNIST",
        epochs=10,
        batch_size=128,
        hidden_dim=64,
        )
if __name__=="__main__":
    main()