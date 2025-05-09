from src.train import train

def main():
    train(
        model_type="DeepSSM",
        dataset="psMNIST",
        epochs=100,
        batch_size=128,
        hidden_dim=256,
        )
if __name__=="__main__":
    main()