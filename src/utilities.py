import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from: https://www.tutorialspoint.com/how-to-plot-csv-data-using-matplotlib-and-pandas-in-python


def main(base_path, batch_per_epoch, info_per_batch=True, gradient_balance=False):


    if gradient_balance:
        headers = ['epoch', 'batch', 'r_loss_real', 'd_loss', 'd_loss_real', 'd_loss_fake', 'g_final_loss',
                   'g_loss', 'r_loss_fake', 'alpha', 'g_loss_std', 'r_loss_fake_std', 'r_loss_balanced']
    else:
        headers = ['epoch', 'batch', 'r_loss_real', 'd_loss', 'd_loss_real', 'd_loss_fake', 'g_final_loss',
                   'g_loss', 'r_loss_fake']

    df = pd.read_csv(os.path.join(base_path, 'batch_summary.csv'))
    print(list(df.columns.values))
    df_mean = df.groupby(np.arange(len(df)) // batch_per_epoch).mean()
    print(df_mean)

    df_mean.plot(x="epoch", y=["d_loss", "d_loss_fake", "d_loss_real"])
    plt.savefig(os.path.join(base_path, 'disc_loss_vis_per_epoch.png'))
    plt.clf()
    if gradient_balance:
        df_mean.plot(x="epoch", y=["r_loss_fake", "g_loss", "r_loss_balanced", "g_final_loss", "r_loss_fake_std", "g_loss_std"])
    else:
        df_mean.plot(x="epoch", y=["r_loss_fake", "g_loss", "g_final_loss"])
    plt.savefig(os.path.join(base_path, 'rec_gen_vis_per_epoch.png'))
    plt.clf()

    if gradient_balance:
        df_mean.plot(x="epoch", y=["r_loss_fake", "r_loss_real", "r_loss_balanced", "r_loss_fake_std", "g_loss_std"])
    else:
        df_mean.plot(x="epoch", y=["r_loss_fake", "r_loss_real"])
    plt.savefig(os.path.join(base_path, 'rec_loss_vis_per_epoch.png'))
    plt.clf()

    # loss/gradient visualisation over batches
    if info_per_batch:
        df = df.astype({'batch': 'int32'})
        print(df["d_loss_fake"].max())
        print(df["d_loss_fake"].mean())
        print(df)
        df.plot(x="batch", y=["d_loss", "d_loss_fake", "d_loss_real"])
        plt.savefig(os.path.join(base_path, 'disc_loss_vis_per_batch.png'))
        plt.clf()
        # if gradient_balance:
        #     df_mean.plot(x="epoch", y=["r_loss_fake", "g_loss", "r_loss_balanced", "g_final_loss", "r_loss_fake_std",
        #                                "g_loss_std"])
        # else:
        #     df_mean.plot(x="epoch", y=["r_loss_fake", "g_loss", "g_final_loss"])
        # plt.savefig(os.path.join(base_path, 'rec_gen_vis_per_epoch.png'))
        # plt.clf()
        #
        # if gradient_balance:
        #     df_mean.plot(x="epoch",
        #                  y=["r_loss_fake", "r_loss_real", "r_loss_balanced", "r_loss_fake_std", "g_loss_std"])
        # else:
        #     df_mean.plot(x="epoch", y=["r_loss_fake", "r_loss_real"])
        # plt.savefig(os.path.join(base_path, 'rec_loss_vis_per_epoch.png'))
        # plt.clf()


if __name__ == '__main__':
    main(base_path='../data/output/ex30/', batch_per_epoch=2512, info_per_batch=True, gradient_balance=True)
