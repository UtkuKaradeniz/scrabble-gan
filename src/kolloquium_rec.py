import sys
sys.path.extend(['/scrabble-gan/'])

import os
import random

import gin
import time
import numpy as np
import tensorflow as tf
import editdistance

from src.bigacgan.arch_ops import spectral_norm
from src.bigacgan.data_utils import load_prepare_data, load_random_word_list, return_stats, write_words
from src.bigacgan.net_architecture import make_recognizer, make_my_recognizer
from src.bigacgan.net_loss import hinge, not_saturating
from src.bigacgan.net_architecture import ctc_loss

gin.external_configurable(hinge)
gin.external_configurable(not_saturating)
gin.external_configurable(spectral_norm)

from src.dinterface.dinterface import init_reading


@gin.configurable
def setup_optimizer(g_lr, d_lr, r_lr, beta_1, beta_2, loss_fn, disc_iters, apply_gradient_balance,
                    gradient_balance_type, rmsprop):
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=beta_1, beta_2=beta_2)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=beta_1, beta_2=beta_2)
    if rmsprop:
        recognizer_optimizer = tf.keras.optimizers.RMSprop(learning_rate=r_lr)
    else:
        recognizer_optimizer = tf.keras.optimizers.Adam(learning_rate=r_lr, beta_1=beta_1, beta_2=beta_2)
    return generator_optimizer, discriminator_optimizer, recognizer_optimizer, loss_fn, disc_iters, \
           apply_gradient_balance, gradient_balance_type


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


def validate_batch(char_vector, labels, decoded):
    num_char_err_vb = 0
    num_char_total = 0

    char_str_wb, char_str_vb = [], []

    # remove CTC-blank label
    decode = [[[int(p) for p in x if p != -1] for x in y] for y in decoded]
    decode = decode[0]

    # convert vanillaBeamSearch vectors to words
    for text in decode:
        # vector to chars and chars to words
        char_str_vb.append(''.join([char_vector[label] for label in text]))

    fake_words = []
    for vector in labels:
        fake_words.append(''.join([char_vector[label] for label in vector]))

    # calculate error rates
    for i in range(len(char_str_vb)):
        dist_vb = editdistance.eval(char_str_vb[i], fake_words[i])
        num_char_err_vb += dist_vb

        num_char_total += len(labels[i])

        print('vanillaBeam : [OK]' if dist_vb == 0 else 'vanillaBeam : [ERR:%d]' % dist_vb, '"' +
              fake_words[i] + '"', '->', '"' + char_str_vb[i] + '"')

    return num_char_err_vb, num_char_total


def validate_recognizer(recognizer, char_vector, valid1_dataset, batch_size, valid1_words):
    # https://github.com/arthurflor23/handwritten-text-recognition/blob/8d9fcd4b4a84e525ba3b985b80954e2170066ae2/src/network/model.py#L435
    """Predict words generated with Generator"""
    num_batch_elements = len(valid1_words) // batch_size

    num_char_err_vb = 0
    num_char_total = 0

    for i in range(num_batch_elements):
        image_batch, label_batch = next(valid1_dataset)
        real_labels = label_batch

        # calculate time-steps
        time_steps = recognizer(image_batch, training=False)

        sequence_length_real = len(real_labels[0])
        inp_len_real = -1 + sequence_length_real * 4
        input_length = np.asarray([inp_len_real for _ in range(len(np.asarray(time_steps)))])
        # decode time-steps
        decoded, _ = tf.keras.backend.ctc_decode(y_pred=time_steps,
                                                 input_length=input_length, greedy=False, beam_width=50)

        char_err_vb, char_total = validate_batch(char_vector, label_batch, decoded)

        num_char_err_vb += char_err_vb
        num_char_total += char_total

    char_error_rate_vb = num_char_err_vb / num_char_total
    print(f'Rec. Character error rate (VanillaBeam Search): {char_error_rate_vb * 100.0}%.')

    return char_error_rate_vb


def validate(recognizer, char_vector, valid1_dataset, batch_size, valid1_words):

    # validate recognizer
    r_err_vb = validate_recognizer(recognizer, char_vector, valid1_dataset, batch_size, valid1_words)

    return r_err_vb


def prepare_gan_input(batch_size, random_words, bucket_size, buckets, bucket_position):
    # prepare Gen. input
    # noise = tf.random.normal([batch_size, latent_dim])
    random_bucket_idx = np.random.choice(buckets, 1)
    random_bucket_idx = random_bucket_idx[0]

    # random_bucket_idx = len(labels[0]) - 1
    if len(random_words) != bucket_size:
        print(len(random_words))
        print(bucket_size)
        print(len(random_words[len(random_words) - 1]))
        print(random.choice(random_words[random_bucket_idx]))
        raise "load_random_word_list not working"

    bucket_length = len(random_words[random_bucket_idx])
    to_check = bucket_position[random_bucket_idx] + batch_size

    # check if bucket has enough elements
    if to_check > bucket_length:
        # remove the entry from buckets, so that it does not get chosen again
        buckets.remove(random_bucket_idx)
        return None, None

    # get the next batch of fake labels from bucket
    sample_idx = bucket_position[random_bucket_idx]
    fake_labels = np.array([random_words[random_bucket_idx][sample_idx + i] for i in range(batch_size)], np.int32)

    # increment last bucket position
    bucket_position[random_bucket_idx] += batch_size

    return -1, fake_labels


def train(train_dataset, valid1_dataset, valid1_words, recognizer, checkpoint, recognizer_optimizer, batch_size,
          epochs, bucket_size, char_vec, random_words, gen_path):
    """
    Whole training procedure

    :param dataset:                     python generator
    :param generator:                   generator model
    :param discriminator:               discriminator model
    :param recognizer:                  Auxiliary Classifier (CRNN)
    :param composite_gan:               Composite GAN
    :param checkpoint:                  tf.train.Checkpoint
    :param checkpoint_prefix:           directory path where to store checkpoints
    :param generator_optimizer:         generator optimizer
    :param discriminator_optimizer:     discriminator optimizer
    :param recognizer_optimizer:        recognizer optimizer
    :param seed_labels:                 noise vector + random labels
    :param buffer_size:                 buffer size (number of training samples)
    :param batch_size:                  batch size
    :param epochs:                      number of epochs
    :param model_path:                  directory path where to store trained model
    :param latent_dim:                  noise vector
    :param gen_path:                    directory path where to store generated images
    :param loss_fn:                     loss function
    :param disc_iters:                  take <disc_iters> D steps per G step
    :param random_words:                list of random words
    :param bucket_size:                 max sequence length
    :param char_vector:                 valid vocabulary represented as array of chars/ string
    :return:
    """
    train_sample_size = sum(len(bucket) for bucket in random_words)
    batch_per_epoch = int(train_sample_size / batch_size) + 1

    print('no. training samples: ', train_sample_size)
    print('batch size:           ', batch_size)
    print('no. batch_per_epoch:  ', batch_per_epoch)
    print('epoch size:           ', epochs)

    best_char_error_rate = float('inf')  # best validation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred

    # define checkpoint save paths
    recognizer_save_dir = os.path.join(checkpoint, 'recognizer/')

    # create folders to save models
    if not os.path.exists(recognizer_save_dir):
        os.makedirs(recognizer_save_dir)

    # create summary files
    batch_summary = open(gen_path + "batch_summary.txt", "w")
    epoch_summary = open(gen_path + "epoch_summary.txt", "w")

    # write headers to files
    header = "r_loss_real"
    epoch_summary.write(header + "cer_r_vb\n")
    batch_summary.write(header + "\n")

    # list to keep track of which buckets have elements and its save
    buckets = []
    buckets_save = []
    # list to keep track of where we are at each bucket and its save
    bucket_position = []
    bucket_position_save = []
    for i in range(0, bucket_size, 1):
        bucket_position.append(0)
        bucket_position_save.append(0)
        buckets.append(i)
        buckets_save.append(i)

    # training loop
    print('training...')
    epoch_idx = 0
    while True:
        start = time.time()

        # variables for total of losses
        r_loss_real_total = 0.0

        for batch_idx in range(batch_per_epoch):
            # get new train batch for Recognizer
            image_batch, label_batch = next(train_dataset)

            # if we run out of buckets, refill them and continue
            if len(buckets) == 0:
                # reset buckets
                buckets = [i for i in buckets_save]
                # reset bucket positions
                bucket_position = [j for j in bucket_position_save]
                continue

            # get noise and fake labels for Generator
            _, fake_labels = prepare_gan_input(batch_size, random_words, bucket_size, buckets, bucket_position)

            # if we ran out of a bucket in random words, choose another
            if fake_labels is None:
                continue

            # training step
            r_loss_real = train_step(epoch_idx, batch_idx, batch_per_epoch, image_batch, label_batch, recognizer,
                                     recognizer_optimizer)

            # write batch summary to file
            batch_summary.write(str(r_loss_real) + '\n')

            # append to lists for epoch summary
            r_loss_real_total += r_loss_real

        # validate Generator and Recognizer with CER
        r_err_vb = validate(recognizer, char_vec, valid1_dataset, batch_size, valid1_words)
        char_err = r_err_vb

        divider = batch_per_epoch
        epoch_summary.write(str(r_loss_real_total / divider) + ";" + str(char_err * 100) + '\n')

        # define sub-folders to save weights after each epoch
        recognizer_epoch_save = os.path.join(recognizer_save_dir, str(epoch_idx + 1) + '/')

        # create folders to save models
        if not os.path.exists(recognizer_epoch_save):
            os.makedirs(recognizer_epoch_save)

        # save recognizer
        recognizer.save_weights(recognizer_epoch_save + 'cktp-' + str(epoch_idx + 1))

        print('Time for epoch {} is {} sec'.format(epoch_idx + 1, time.time() - start))

        # if best validation accuracy so far, save model parameters
        if char_err < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_err
            no_improvement_since = 0
        else:
            print(f'Character error rate not improved, best so far: {char_err}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= 15:
            print(f'No more improvement since {15} epochs. Training stopped.')
            break

    batch_summary.close()
    epoch_summary.close()


def train_step(epoch_idx, batch_idx, batch_per_epoch, images, labels, recognizer, recognizer_optimizer):
    """
    Single training loop

    :param epoch_idx:                   current epoch idx
    :param batch_idx:                   current batch idx
    :param batch_per_epoch:             number of batches per epoch
    :param images:                      batch of images
    :param labels:                      batch of labels
    :param discriminator:               discriminator model
    :param recognizer:                  Auxiliary Classifier (CRNN)
    :param composite_gan:               Composite GAN
    :param generator_optimizer:         generator optimizer
    :param discriminator_optimizer:     discriminator optimizer
    :param recognizer_optimizer:        recognizer optimizer
    :param batch_size:                  batch size
    :param latent_dim:                  noise vector
    :param loss_fn:                     loss function (gin configurable)
    :param disc_iters:                  take <disc_iters> D steps per G step
    :param random_words:                list of random words
    :param bucket_size:                 max sequence length
    :return:
    """

    # obtain shapes
    batch_size_real = images.shape[0]
    sequence_length_real = len(labels[0])

    # compute loss & update gradients
    with tf.GradientTape() as rec_tape:

        # compute R(real)
        inp_len_real = -1 + sequence_length_real * 4
        time_steps = recognizer(images, training=True)

        r_real_logits = ctc_loss(labels, time_steps, np.array([[inp_len_real]] * batch_size_real),
                                 np.array([[sequence_length_real]] * batch_size_real))

        # compute stats
        r_loss_real_mean = tf.reduce_mean(r_real_logits)

    tf.print('>%d, %d/%d, r_loss_real=%.3f' % (epoch_idx + 1, batch_idx + 1, batch_per_epoch, r_loss_real_mean))

    # compute and apply gradients of D and R
    recognizer.trainable = True
    gradients_of_recognizer = rec_tape.gradient(r_real_logits, recognizer.trainable_variables)
    recognizer_optimizer.apply_gradients(zip(gradients_of_recognizer, recognizer.trainable_variables))

    return r_loss_real_mean.numpy()


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

    # for testing in windows
    if os.name == 'nt':
        read_dir = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\IAM_mygan\\words-Reading'
        test_dir = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\IAM_mygan\\words-Reading-test\\'
        raw_dir = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\IAM_mygan\\img'
        gen_path = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\output\\ex30'

    train_dir, valid1_dir, valid2_dir = create_database_dirs(raw_dir, ckpt_path, m_path, gen_path, read_dir,
                                                             in_dim, bucket_size)

    # load random words into memory (used for word generation by G)
    random_words = load_random_word_list(read_dir, bucket_size, char_vec)

    # print dataset info
    _, valid1_words, _ = dataset_stats(train_dir, valid1_dir, valid2_dir, bucket_size, random_words)

    # load and preprocess dataset (python generator)
    train_dataset = load_prepare_data(in_dim, batch_size, train_dir, char_vec, bucket_size)
    valid1_dataset = load_prepare_data(in_dim, batch_size, valid1_dir, char_vec, bucket_size)

    # initialize recognizer
    if my_rec:
        print("using my recognizer")
        recognizer = make_my_recognizer(in_dim, n_classes + 1, restore=False)
    else:
        print("using scrabbleGAN recognizer")
        recognizer = make_recognizer(in_dim, n_classes + 1)

    _, _, recognizer_optimizer, _, _, _, _ = setup_optimizer()

    train(train_dataset, valid1_dataset, valid1_words, recognizer, ckpt_path, recognizer_optimizer, batch_size,
          epochs, bucket_size, char_vec, random_words, gen_path)


if __name__ == "__main__":
    main()
