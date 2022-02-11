import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from: https://www.tutorialspoint.com/how-to-plot-csv-data-using-matplotlib-and-pandas-in-python


def main():
    plt.rcParams["figure.autolayout"] = True

    gradient_balance = True
    base_path = '../data/output/ex30/'
    batch_per_epoch = 2512

    if gradient_balance:
        headers = ['epoch', 'batch', 'r_loss_real', 'd_loss', 'd_loss_real', 'd_loss_fake', 'g_final_loss',
                   'g_loss', 'r_loss_fake', 'alpha', 'g_loss_std', 'r_loss_fake_std', 'r_loss_balanced']
    else:
        headers = ['epoch', 'batch', 'r_loss_real', 'd_loss', 'd_loss_real', 'd_loss_fake', 'g_final_loss',
                   'g_loss', 'r_loss_fake']

    df = pd.read_csv(os.path.join(base_path, 'batch_summary.csv'))
    df2 = df.groupby(np.arange(len(df)) // batch_per_epoch).mean()
    df2.plot(x="epoch", y=["d_loss", "d_loss_fake", "d_loss_real"])
    plt.show()
    plt.clf()
    df2.plot(x="epoch", y=["r_loss_fake", "r_loss_balanced", 'g_loss_std', 'r_loss_fake_std', 'alpha'])
    plt.show()
    plt.clf()
    df2.plot(x="epoch", y=["r_loss_fake", "r_loss_real"])
    plt.show()
    plt.clf()
    df2.plot(x="epoch", y=["d_loss", "g_loss", "r_loss_real"])
    plt.show()
    plt.clf()

if __name__ == '__main__':
    main()