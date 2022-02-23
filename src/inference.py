import sys
sys.path.extend(['/scrabble-gan/'])

import os
import random

import gin
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

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


def load_test_data(input_dim, batch_size, reading_dir, char_vector, bucket_size):
    """
    load_prepare_data from dinterface.load_prepare_data configured to go through the entire dataset, no matter the
    bucket size

    (1) read buckets into memory
    (2) create python generator

    :param input_dim:
    :param batch_size:
    :param reading_dir:
    :param char_vector:
    :param bucket_size:
    :return:
    """

    h, w, c = input_dim

    data_buckets = {}
    number_samples = 0

    # (1) read buckets into memory
    for i in range(1, bucket_size + 1, 1):

        imgs = []
        labels = []

        reading_dir_bucket = reading_dir + str(i) + '/'
        file_list = os.listdir(reading_dir_bucket)
        file_list = [fi for fi in file_list if fi.endswith(".txt")]

        for file in file_list:
            with open(reading_dir_bucket + file, 'r', encoding='utf8') as f:
                # 'auto' -> [0, 20, 19, 14]
                label = [char_vector.index(char) for char in f.readline()]
                img = cv2.imread(os.path.join(reading_dir_bucket, os.path.splitext(file)[0] + '.png'), 0)
                imgs.append(img)
                labels.append(label)
                number_samples += 1

        # shuffle lists
        shuffled = list(zip(imgs, labels))
        random.shuffle(shuffled)
        imgs_shuffled, labels_shuffled = zip(*shuffled)
        data_buckets[i] = (imgs_shuffled, labels_shuffled)

    # list to keep track of which buckets have elements and its save
    buckets = []
    buckets_save = []
    # list to keep track of where we are at each bucket and its save
    bucket_position = []
    bucket_position_save = []
    for i in range(1, bucket_size + 1, 1):
        bucket_position.append(0)
        bucket_position_save.append(0)
        buckets.append(i)
        buckets_save.append(i)

    # (2) create python generator
    while True:
        final_batch_size = None

        # select random bucket
        random_bucket_idx = np.random.choice(buckets, 1)
        random_bucket_idx = int(random_bucket_idx[0])

        bucket_length = len(data_buckets[random_bucket_idx][1])
        to_check = bucket_position[random_bucket_idx-1] + batch_size
        # check if bucket has enough elements
        if to_check > bucket_length:
            # remove the entry from buckets, so that it does not get chosen again
            buckets.remove(random_bucket_idx)
            # put the remaining words into a smaller batch
            if bucket_length - bucket_position[random_bucket_idx-1] < batch_size:
                final_batch_size = len(data_buckets[random_bucket_idx][0]) - bucket_position[random_bucket_idx-1]

        image_batch = []
        label_batch = []

        sample_idx = bucket_position[random_bucket_idx-1]
        loop_iter = final_batch_size if final_batch_size is not None else batch_size
        for i in range(loop_iter):
            # retrieve samples from bucket of size batch_size
            image_batch.append(data_buckets[random_bucket_idx][0][sample_idx + i])
            label_batch.append(data_buckets[random_bucket_idx][1][sample_idx + i])
        bucket_position[random_bucket_idx-1] += loop_iter

        # convert to numpy array
        image_batch = np.array(image_batch).astype('float32')
        label_batch = np.array(label_batch).astype(np.int32)

        # normalize images to [-1, 1]
        image_batch = image_batch.reshape(-1, h, int((h / 2) * random_bucket_idx), c)
        image_batch = (image_batch - 127.5) / 127.5

        yield (image_batch, label_batch)


def dataset_stats(test_dir, bucket_size):
    # count test sample size and print it
    test_words = return_stats(test_dir, bucket_size)
    print(f'no. test samples: {len(test_words)}\n')

    # write the words into a separate files for future use
    write_words(test_dir, test_words, mode='test')

    return test_words


def validate_batch(char_vector, labels, time_steps, decoded, wbs):
    num_char_err_vb = 0
    # num_char_err_wb = 0
    num_char_total = 0

    # wbs_in = tf.transpose(time_steps, perm=[1, 0, 2])
    # # decode time-steps with WordBeamSearch
    # label_str = wbs.compute(wbs_in)

    char_str_wb, char_str_vb = [], []

    # remove CTC-blank label
    decode = [[[int(p) for p in x if p != -1] for x in y] for y in decoded]
    decode = decode[0]

    # convert vanillaBeamSearch vectors to words
    for text in decode:
        # vector to chars and chars to words
        char_str_vb.append(''.join([char_vector[label] for label in text]))

    # # convert WordBeamSearch vectors to words
    # for curr_label_str in label_str:
    #     char_str_wb.append(''.join([char_vector[label] for label in curr_label_str]))
    #
    # assert len(char_str_vb) == len(char_str_wb)

    fake_words = []
    for vector in labels:
        fake_words.append(''.join([char_vector[label] for label in vector]))

    # calculate error rates
    for i in range(len(char_str_vb)):
        dist_vb = editdistance.eval(char_str_vb[i], fake_words[i])
        num_char_err_vb += dist_vb

        # dist_wb = editdistance.eval(char_str_wb[i], fake_words[i])
        # num_char_err_wb += dist_wb

        num_char_total += len(labels[i])

        print('vanillaBeam : [OK]' if dist_vb == 0 else 'vanillaBeam : [ERR:%d]' % dist_vb, '"' +
              fake_words[i] + '"', '->', '"' + char_str_vb[i] + '"')
        # print('wordBeam : [OK]' if dist_wb == 0 else 'wordBeam : [ERR:%d]' % dist_wb, '"' + fake_words[i]
        #       + '"', '->', '"' + char_str_wb[i] + '"')

    return num_char_err_vb, num_char_total


def validate_recognizer(recognizer, char_vector, valid1_dataset, batch_size, valid1_words, wbs):
    # https://github.com/arthurflor23/handwritten-text-recognition/blob/8d9fcd4b4a84e525ba3b985b80954e2170066ae2/src/network/model.py#L435
    """Predict words generated with Generator"""
    num_batch_elements = len(valid1_words) // batch_size

    num_char_err_vb = 0
    num_char_err_wb = 0
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

        char_err_vb, char_total = validate_batch(char_vector, label_batch, time_steps, decoded, -1)

        num_char_err_vb += char_err_vb
        # num_char_err_wb += char_err_wb
        num_char_total += char_total

    char_error_rate_vb = num_char_err_vb / num_char_total
    # char_error_rate_wb = num_char_err_wb / num_char_total
    print(f'Rec. Character error rate (VanillaBeam Search): {char_error_rate_vb * 100.0}%.')

    return char_error_rate_vb


def validate_generator(generator, recognizer, char_vector, valid2_dataset, batch_size, latent_dim, valid2_words):
    # https://github.com/arthurflor23/handwritten-text-recognition/blob/8d9fcd4b4a84e525ba3b985b80954e2170066ae2/src/network/model.py#L435
    """Predict words generated with Generator"""
    num_batch_elements = len(valid2_words) // batch_size

    num_char_err_vb = 0
    # num_char_err_wb = 0
    num_char_total = 0

    for i in range(num_batch_elements):
        _, label_batch = next(valid2_dataset)

        # generate latent points + take labels form valid2 dataset
        noise = tf.random.normal([batch_size, latent_dim])
        fake_labels = label_batch

        # generate fake images
        generated_imgs = generator([noise, fake_labels], training=False)

        # predict images / calculate time-steps
        time_steps = recognizer(generated_imgs, training=False)

        sequence_length_fake = len(fake_labels[0])
        inp_len_fake = -1 + sequence_length_fake * 4
        input_length = np.asarray([inp_len_fake for _ in range(len(np.asarray(time_steps)))])
        # decode time-steps with vanillaBeamSearch
        decoded, _ = tf.keras.backend.ctc_decode(y_pred=time_steps, input_length=input_length, greedy=False,
                                                 beam_width=50)

        char_err_vb, char_total = validate_batch(char_vector, label_batch, time_steps, decoded, -1)

        num_char_err_vb += char_err_vb
        # num_char_err_wb += char_err_wb
        num_char_total += char_total

    # print validation result
    char_error_rate_vb = num_char_err_vb / num_char_total
    # char_error_rate_wb = num_char_err_wb / num_char_total
    print(f'Gen. Character error rate (VanillaBeam Search): {char_error_rate_vb * 100.0}%. ')

    return char_error_rate_vb


def test(generator, recognizer, char_vector, valid2_dataset, batch_size, latent_dim, valid2_words):
    # https://github.com/arthurflor23/handwritten-text-recognition/blob/8d9fcd4b4a84e525ba3b985b80954e2170066ae2/src/network/model.py#L435
    """Predict words generated with Generator"""
    num_batch_elements = len(valid2_words) // batch_size

    num_char_err_vb = 0
    # num_char_err_wb = 0
    num_char_total = 0

    for i in range(num_batch_elements):
        _, label_batch = next(valid2_dataset)

        # generate latent points + take labels form valid2 dataset
        noise = tf.random.normal([batch_size, latent_dim])
        fake_labels = label_batch

        # generate fake images
        generated_imgs = generator([noise, fake_labels], training=False)

        # predict images / calculate time-steps
        time_steps = recognizer(generated_imgs, training=False)

        sequence_length_fake = len(fake_labels[0])
        inp_len_fake = -1 + sequence_length_fake * 4
        input_length = np.asarray([inp_len_fake for _ in range(len(np.asarray(time_steps)))])
        # decode time-steps with vanillaBeamSearch
        decoded, _ = tf.keras.backend.ctc_decode(y_pred=time_steps, input_length=input_length, greedy=False,
                                                 beam_width=50)

        char_err_vb, char_total = validate_batch(char_vector, label_batch, time_steps, decoded, -1)

        num_char_err_vb += char_err_vb
        # num_char_err_wb += char_err_wb
        num_char_total += char_total

    # print validation result
    char_error_rate_vb = num_char_err_vb / num_char_total
    # char_error_rate_wb = num_char_err_wb / num_char_total
    print(f'Gen. Character error rate (VanillaBeam Search): {char_error_rate_vb * 100.0}%. ')

    return char_error_rate_vb


def test(gen_scrabble_rec, gen_my_rec, scrabble_rec, my_rec):
    # test my recognizer
    my_rec_err = test_rec(my_rec, char_vector, valid1_dataset, batch_size, valid1_words)
    # test scrabble recognizer
    s_rec_err = test_rec(gen_my_rec, char_vector, valid1_dataset, batch_size, valid1_words)

    # cross-test generators with recognizers
    # generator trained with my recognizer tested with my recognizer
    my_gen_my_rec_err = test_gen(gen_my_rec, gen_my_rec)
    # generator trained with my recognizer tested with scrabbleGAN recognizer
    my_gen_s_rec_err = test_gen(gen_my_rec, scrabble_rec)
    # generator trained with scarabbleGAN recognizer tested with my recognizer
    s_gen_my_rec_err = test_gen(gen_scrabble_rec, my_rec)
    # generator trained with scrabbleGAN recognizer tested with scrabbleGAN recognizer
    s_gen_s_rec_err = test_gen(gen_scrabble_rec, scrabble_rec)

    return my_rec_err, s_rec_err, my_gen_my_rec_err, my_gen_s_rec_err, s_gen_my_rec_err, s_gen_s_rec_err


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
        test_dir = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\IAM_mygan\\words-Reading-test\\'
        raw_dir = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\IAM_mygan\\img'
        gen_path = 'C:\\Users\\tuk\\Documents\\Uni-Due\\Bachelorarbeit\\dir_working\\scrabble-gan\\data\\output\\ex30'

    test_dir = read_dir + '-' + 'test' + '/'
    # convert test dataset to GAN format
    if not os.path.exists(test_dir):
        init_reading(raw_dir, read_dir, in_dim, bucket_size, mode='test')
    # create output directory
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)

    # load and preprocess dataset (python generator)
    test_dataset = load_prepare_data(in_dim, batch_size, test_dir, char_vec, bucket_size)

    # print dataset info
    test_words = dataset_stats(test_dir, bucket_size)

    num_batch_elements = len(test_words) // batch_size
    print("number of test words: ", len(test_words))
    print("batch_size: ", batch_size)
    print("num_batch_elements: ", num_batch_elements)
    counter = 0
    while True:
        img, label = next(test_dataset)
        print("label length: ", len(label))
        print(counter)
        counter += 1

    # initialize two generators, one scrabbleGAN recognizer and one myRecognizer for testing
    gen_1 = make_generator(latent_dim, in_dim, embed_y, kernel_reg, g_bw_attention, n_classes)
    gen_2 = make_generator(latent_dim, in_dim, embed_y, kernel_reg, g_bw_attention, n_classes)
    scrabble_rec = make_recognizer(in_dim, n_classes + 1)
    my_rec = make_my_recognizer(in_dim, n_classes + 1, restore=False)

    # load pre-trained models (gen_1 -> generator trained with scrabble_rec, gen_2 -> generator trained with my_rec)
    gen_scrabble_rec, gen_my_rec, scrabble_rec, my_rec = load_weights(gen_1, gen_2, scrabble_rec, my_rec)

    # inference function
    test(gen_scrabble_rec, gen_my_rec, scrabble_rec, my_rec, test_dataset)


if __name__ == "__main__":
    main()
