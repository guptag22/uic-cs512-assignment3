import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_accuracy(epoch, train, test, file_name):
# Data
    x= [x for x in range(epoch)]
    w = 10
    h = 8
    d = 100
    path = "../Figures/"
    plt.legend()
    plt.figure(figsize=(w, h), dpi=d)
    plt.plot(x, train, color='red', linewidth=2, label="Train")
    plt.plot(x, test, color = 'blue', linewidth = 2, linestyle = 'dashed', label='Test')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
    legend.get_frame()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(path + file_name)

