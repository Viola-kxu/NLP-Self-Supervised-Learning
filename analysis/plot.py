import matplotlib.pyplot as plt
import numpy as np

losses = [1.3718, 1.2748, 1.2377, 1.1978, 1.2077]
eval_losses = [1.4799555540084839, 1.434308648109436, 1.45461106300354, 1.4208133220672607, 1.420228910446167]
losses_og = [1.4007, 1.377, 1.351, 1.3562, 1.372186737060547]
eval_losses_og = [1.3945014476776123, 1.3862303495407104, 1.3861638307571411, 1.39227294921875, 1.3952981233596802]
eval_accurs = [0.32, 0.35, 0.31, 0.29, 0.33]
eval_accurs_og = [0.23, 0.25, 0.25, 0.23, 0.26]

if __name__ == "__main__":
    plt.plot(np.linspace(1, 5, 5), eval_accurs, label="With generated data")
    plt.plot(np.linspace(1, 5, 5), eval_accurs_og, label="Without generated data")
    plt.hlines(0.4, 1, 5, colors='g', alpha=0.5, linestyles='dashed', label="Baseline (GPT3.5)")
    plt.hlines(0.25, 1, 5, colors='r', alpha=0.5, linestyles='dashed')
    plt.title("Evaluation Accuracy on BERT Using Generated Data")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0.15, 0.45)
    plt.legend()
    plt.show()
    plt.savefig("val_acc.png")