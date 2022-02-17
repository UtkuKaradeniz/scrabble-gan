import sys
sys.path.extend(['/scrabble-gan/'])

import os
import random

import gin
import numpy as np
import tensorflow as tf

from src.bigacgan.arch_ops import spectral_norm
from src.bigacgan.data_utils import load_prepare_data, train, make_gif, load_random_word_list, return_sample_size
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
def setup_io(base_path, checkpoint_dir, gen_imgs_dir, model_dir, raw_dir, read_dir, input_dim, n_classes, seq_len,
             char_vec, bucket_size, mode):
    gen_path = base_path + gen_imgs_dir
    ckpt_path = base_path + checkpoint_dir
    m_path = base_path + model_dir
    raw_dir = base_path + raw_dir
    read_dir = base_path + read_dir + '-' + mode + '/'
    return input_dim, n_classes, seq_len, bucket_size, ckpt_path, gen_path, m_path, raw_dir, read_dir, char_vec, mode


def main():
    # init params
    gin.parse_config_file('scrabble_gan.gin')
    epochs, batch_size, latent_dim, embed_y, num_style, num_gen, kernel_reg, g_bw_attention, d_bw_attention, \
    my_rec, my_disc = get_shared_specs()
    in_dim, n_classes, seq_len, bucket_size, ckpt_path, gen_path, m_path, raw_dir, read_dir, char_vec, mode = setup_io()

    # convert IAM Handwriting dataset (words) to GAN format
    if not os.path.exists(read_dir):
        print('converting iamDB-Dataset to GAN format...')
        init_reading(raw_dir, read_dir, in_dim, bucket_size, mode)

    # load random words into memory (used for word generation by G)
    random_words = load_random_word_list(read_dir, bucket_size, char_vec)

    # load and preprocess dataset (python generator)
    train_dataset = load_prepare_data(in_dim, batch_size, read_dir, char_vec, bucket_size)
    buf_size = return_sample_size(reading_dir=read_dir, bucket_size=bucket_size)

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
        recognizer = make_my_recognizer(in_dim, seq_len, n_classes + 1, restore=False)
    else:
        print("using scrabbleGAN recognizer")
        recognizer = make_recognizer(in_dim, seq_len, n_classes + 1)

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

    train(train_dataset, generator, discriminator, recognizer, gan, ckpt_path, -1, generator_optimizer,
          discriminator_optimizer, recognizer_optimizer, [seeds, labels], buf_size, batch_size, epochs, m_path,
          latent_dim, gen_path, loss_fn, disc_iters, apply_gradient_balance, random_words, bucket_size, char_vec)

    # use imageio to create an animated gif using the images saved during training.
    make_gif(gen_path)


if __name__ == "__main__":
    main()
