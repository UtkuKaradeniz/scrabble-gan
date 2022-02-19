import sys
sys.path.extend(['/scrabble-gan/'])

import os
import random

import gin
import numpy as np
import tensorflow as tf
import matplotlib as plt

from src.bigacgan.arch_ops import spectral_norm
from src.bigacgan.data_utils import load_prepare_data, train, make_gif, load_random_word_list, return_stats, write_words
from src.bigacgan.net_architecture import make_generator, make_discriminator, make_my_discriminator
from src.bigacgan.net_architecture import make_recognizer, make_my_recognizer, make_gan
from src.bigacgan.net_loss import hinge, not_saturating

gin.external_configurable(hinge)
gin.external_configurable(not_saturating)
gin.external_configurable(spectral_norm)

from src.dinterface.dinterface import init_reading

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@gin.configurable
def setup_optimizer(g_lr, d_lr, r_lr, beta_1, beta_2, loss_fn, disc_iters, apply_gradient_balance, rmsprop):
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=beta_1, beta_2=beta_2)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=beta_1, beta_2=beta_2)
    if rmsprop:
        recognizer_optimizer = tf.keras.optimizers.RMSprop(learning_rate=r_lr)
    else:
        recognizer_optimizer = tf.keras.optimizers.Adam(learning_rate=r_lr, beta_1=beta_1, beta_2=beta_2)
    return generator_optimizer, discriminator_optimizer, recognizer_optimizer, loss_fn, disc_iters, \
           apply_gradient_balance


@gin.configurable('shared_specs')
def get_shared_specs(epochs, batch_size, latent_dim, embed_y, num_style, num_gen, kernel_reg, g_bw_attention,
                     d_bw_attention, my_rec, my_disc):
    return epochs, batch_size, latent_dim, embed_y, num_style, num_gen, kernel_reg, g_bw_attention, d_bw_attention, \
           my_rec, my_disc


@gin.configurable('io')
def setup_io(ex_id, base_path, checkpoint_dir, gen_imgs_dir, model_dir, raw_dir, read_dir, input_dim, n_classes,
             seq_len, char_vec, bucket_size):
    gen_path = base_path + gen_imgs_dir + ex_id
    ckpt_path = base_path + checkpoint_dir + ex_id
    m_path = base_path + model_dir + ex_id
    raw_dir = base_path + raw_dir
    read_dir = base_path + read_dir

    return input_dim, n_classes, seq_len, bucket_size, ckpt_path, gen_path, m_path, raw_dir, read_dir, char_vec


def create_database_dirs(raw_dir, ckpt_path, m_path, gen_path, read_dir, in_dim, bucket_size):
    """creates directories for the current experiment and the database folders if any of them are missing"""
    # create other directories for the current experiment
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(m_path):
        os.makedirs(m_path)
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)

    # create database directories
    train_dir = read_dir + '-' + 'train' + '/'
    valid1_dir = read_dir + '-' + 'valid1' + '/'
    valid2_dir = read_dir + '-' + 'valid2' + '/'

    # convert IAM Handwriting dataset (words) to GAN format
    if not os.path.exists(train_dir):
        init_reading(raw_dir, read_dir, in_dim, bucket_size, mode='train')
    if not os.path.exists(valid1_dir):
        init_reading(raw_dir, read_dir, in_dim, bucket_size, mode='valid1')
    if not os.path.exists(valid2_dir):
        init_reading(raw_dir, read_dir, in_dim, bucket_size, mode='valid2')

    return train_dir, valid1_dir, valid2_dir


def dataset_stats(train_dir, valid1_dir, valid2_dir, bucket_size, random_words):
    # count sample size and print it
    train_words = return_stats(train_dir, bucket_size)
    valid1_words = return_stats(valid1_dir, bucket_size)
    valid2_words = return_stats(valid2_dir, bucket_size)

    print(f'no. train samples: {len(train_words)}\n'
          f'no. valid1 samples: {len(valid1_words)}\n'
          f'no. valid2 samples: {len(valid2_words)}\n'
          f'no. random word samples: {len(random_words)}\n')

    # write only the words into separate files for future use
    write_words(train_dir, train_words, mode='train')
    write_words(valid1_dir, valid1_words, mode='valid1')
    write_words(valid2_dir, valid2_words, mode='valid2')

    return train_words, valid1_words, valid2_words


def main():
    # init params
    gin.parse_config_file('scrabble_gan.gin')
    epochs, batch_size, latent_dim, embed_y, num_style, num_gen, kernel_reg, g_bw_attention, d_bw_attention, \
    my_rec, my_disc = get_shared_specs()
    in_dim, n_classes, seq_len, bucket_size, ckpt_path, gen_path, m_path, raw_dir, read_dir, char_vec = setup_io()

    # TODO: solve path issues in windows
    # for testing in windows
    if os.name == 'nt':
        read_dir = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\IAM_mygan\\words-Reading'
        train_dir = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\IAM_mygan\\words-Reading-train\\'
        valid1_dir = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\IAM_mygan\\words-Reading-valid1\\'
        valid2_dir = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\IAM_mygan\\words-Reading-valid2\\'
        raw_dir = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\IAM_mygan\\img'
        gen_path = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\output\\ex30'

    train_dir, valid1_dir, valid2_dir = create_database_dirs(raw_dir, ckpt_path, m_path, gen_path, read_dir,
                                                             in_dim, bucket_size)

    # load random words into memory (used for word generation by G)
    random_words = load_random_word_list(read_dir, bucket_size, char_vec)

    # load and preprocess dataset (python generator)
    train_dataset = load_prepare_data(in_dim, batch_size, train_dir, char_vec, bucket_size)
    while True:
        imgs, labels = next(train_dataset)
        print(len(imgs))
        print(labels)
        fig, ax = plt.subplots(5, 5)
        for k in range(5):
            for j in range(5):
                ax[k, j].imshow(imgs[k * 5 + j], cmap='gray')
                ax[k, j].text(-5, -10, labels[k * 5 + j], fontsize='xx-small')
                print(labels[:5])
        exit(-1)
    valid1_dataset = load_prepare_data(in_dim, batch_size, valid1_dir, char_vec, bucket_size)
    valid2_dataset = load_prepare_data(in_dim, batch_size, valid2_dir, char_vec, bucket_size)

    # print dataset info
    train_words, valid1_words, valid2_words = dataset_stats(train_dir, valid1_dir, valid2_dir, bucket_size, random_words)

    # init generator, discriminator and recognizer
    generator = make_generator(latent_dim, in_dim, embed_y, kernel_reg, g_bw_attention, n_classes)
    if my_disc:
        print("using my discriminator")
        discriminator = make_my_discriminator(in_dim, kernel_reg)
    else:
        print("using scrabbleGAN discriminator")
        discriminator = make_discriminator(in_dim, kernel_reg, d_bw_attention)
    if my_rec:
        print("using my recognizer")
        recognizer = make_my_recognizer(in_dim, n_classes + 1, restore=False)
    else:
        print("using scrabbleGAN recognizer")
        recognizer = make_recognizer(in_dim, n_classes + 1)

    # build composite model (update G through composite model)
    gan = make_gan(generator, discriminator, recognizer)

    # init optimizer for both generator, discriminator and recognizer
    generator_optimizer, discriminator_optimizer, recognizer_optimizer, loss_fn, disc_iters, apply_gradient_balance = setup_optimizer()

    ### choose seed and labels to generate images
    # generate as many styles as needed
    seeds = [tf.random.normal([1, latent_dim]) for _ in range(num_style)]
    # choose random words with random lengths
    random_bucket_idx = np.random.randint(low=3, high=bucket_size, size=num_gen)
    labels = [random.choice(random_words[random_bucket_idx[i]]) for i in range(num_gen)]

    train(train_dataset, valid1_dataset, valid2_dataset, generator, discriminator, recognizer, gan, ckpt_path,
          generator_optimizer, discriminator_optimizer, recognizer_optimizer, [seeds, labels], batch_size, epochs,
          latent_dim, gen_path, loss_fn, disc_iters, apply_gradient_balance, random_words, bucket_size, char_vec,
          train_words, valid1_words, valid2_words)

    # use imageio to create an animated gif using the images saved during training.
    make_gif(gen_path)


if __name__ == "__main__":
    main()
