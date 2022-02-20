import glob
import os
import time

import cv2
import numpy as np
import tensorflow as tf
import editdistance

import imageio
import matplotlib.pyplot as plt
import random
from word_beam_search import WordBeamSearch
from src.bigacgan.net_architecture import ctc_loss


def write_words(reading_dir, words, mode):
    f = open(reading_dir + mode + '-words.txt', 'w', encoding='utf8')
    no_duplicates = list(dict.fromkeys(words))
    for word in no_duplicates:
        f.write("%s\n" % word)
    f.close()


def return_stats(reading_dir, bucket_size):
    words = []
    number_samples = 0

    # (1) read buckets into memory
    for i in range(1, bucket_size + 1, 1):
        reading_dir_bucket = os.path.join(reading_dir, str(i) + '/')
        file_list = os.listdir(reading_dir_bucket)
        file_list = [fi for fi in file_list if fi.endswith(".txt")]
        for file in file_list:
            with open(reading_dir_bucket + file, 'r', encoding='utf8') as f:
                words.append(f.readline())

        number_samples += len(file_list)

    assert number_samples == len(words)

    return words


def load_prepare_data(input_dim, batch_size, reading_dir, char_vector, bucket_size):
    """
    load data into tensor (python generator)

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
        # if we run out of buckets, refill them
        if len(buckets) == 0:
            # reset buckets
            buckets = [i for i in buckets_save]
            # reset bucket positions
            bucket_position = [j for j in bucket_position_save]
            # re-shuffle dictionary
            list_buckets = list(data_buckets.items())
            random.shuffle(list_buckets)
            data_buckets = dict(list_buckets)

        # select random bucket
        random_bucket_idx = np.random.choice(buckets, 1)
        random_bucket_idx = int(random_bucket_idx[0])

        bucket_length = len(data_buckets[random_bucket_idx][1])
        to_check = bucket_position[random_bucket_idx-1] + batch_size
        # check if bucket has enough elements
        if to_check > bucket_length:
            # remove the entry from buckets, so that it does not get chosen again
            buckets.remove(random_bucket_idx)
            continue
            # # if the remaining words are less than 1/4th of the batch size, skip it
            # if bucket_length - bucket_position[random_bucket_idx-1] < batch_size // 4:
            #     continue
            # final_batch_size = len(data_buckets[random_bucket_idx][0]) - bucket_position[random_bucket_idx-1]

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


def validate_batch(char_vector, labels, time_steps, decoded, wbs):
    num_char_err_vb = 0
    num_char_err_wb = 0
    num_char_total = 0

    wbs_in = tf.transpose(time_steps, perm=[1, 0, 2])
    # decode time-steps with WordBeamSearch
    label_str = wbs.compute(wbs_in)

    char_str_wb, char_str_vb = [], []

    # remove CTC-blank label
    decode = [[[int(p) for p in x if p != -1] for x in y] for y in decoded]
    decode = decode[0]

    # convert vanillaBeamSearch vectors to words
    for text in decode:
        # vector to chars and chars to words
        char_str_vb.append(''.join([char_vector[label] for label in text]))

    # convert WordBeamSearch vectors to words
    for curr_label_str in label_str:
        char_str_wb.append(''.join([char_vector[label] for label in curr_label_str]))

    assert len(char_str_vb) == len(char_str_wb)

    fake_words = []
    for vector in labels:
        fake_words.append(''.join([char_vector[label] for label in vector]))

    # calculate error rates
    for i in range(len(char_str_vb)):
        dist_vb = editdistance.eval(char_str_vb[i], fake_words[i])
        num_char_err_vb += dist_vb

        dist_wb = editdistance.eval(char_str_wb[i], fake_words[i])
        num_char_err_wb += dist_wb

        num_char_total += len(labels[i])

        print('vanillaBeam : [OK]' if dist_vb == 0 else 'vanillaBeam : [ERR:%d]' % dist_vb, '"' +
              fake_words[i] + '"', '->', '"' + char_str_vb[i] + '"')
        print('wordBeam : [OK]' if dist_wb == 0 else 'wordBeam : [ERR:%d]' % dist_wb, '"' + fake_words[i]
              + '"', '->', '"' + char_str_wb[i] + '"')

    return num_char_err_vb, num_char_err_wb, num_char_total


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

        char_err_vb, char_err_wb, char_total = validate_batch(char_vector, label_batch, time_steps, decoded, wbs)

        num_char_err_vb += char_err_vb
        num_char_err_wb += char_err_wb
        num_char_total += char_total

    char_error_rate_vb = num_char_err_vb / num_char_total
    char_error_rate_wb = num_char_err_wb / num_char_total
    print(f'Rec. Character error rate (VanillaBeam Search): {num_char_err_vb * 100.0}%. '
          f'Rec. Character error rate (WordBeam Search): {char_error_rate_wb * 100.0}%')

    return char_error_rate_vb, char_error_rate_wb


def validate_generator(generator, recognizer, char_vector, valid2_dataset, batch_size, latent_dim, valid2_words, wbs):
    # https://github.com/arthurflor23/handwritten-text-recognition/blob/8d9fcd4b4a84e525ba3b985b80954e2170066ae2/src/network/model.py#L435
    """Predict words generated with Generator"""
    num_batch_elements = len(valid2_words) // batch_size

    num_char_err_vb = 0
    num_char_err_wb = 0
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

        char_err_vb, char_err_wb, char_total = validate_batch(char_vector, label_batch, time_steps, decoded, wbs)

        num_char_err_vb += char_err_vb
        num_char_err_wb += char_err_wb
        num_char_total += char_total

    # print validation result
    char_error_rate_vb = num_char_err_vb / num_char_total
    char_error_rate_wb = num_char_err_wb / num_char_total
    print(f'Gen. Character error rate (VanillaBeam Search): {num_char_err_vb * 100.0}%. '
          f'Gen. Character error rate (WordBeam Search): {char_error_rate_wb * 100.0}%')

    return char_error_rate_vb, char_error_rate_wb


def validate(generator, recognizer, char_vector, valid1_dataset, valid2_dataset, batch_size, latent_dim,
             valid1_words, valid2_words, train_words):
    chars = word_chars = char_vector
    corpus = ' '.join(valid1_words + valid2_words + train_words)
    wbs = WordBeamSearch(50, 'NGrams', 0.0, corpus.encode('utf8'), chars.encode('utf8'), word_chars.encode('utf8'))

    # validate recognizer
    r_err_vb, r_err_wb = validate_recognizer(recognizer, char_vector, valid1_dataset, batch_size, valid1_words, wbs)

    # validate generator
    g_err_vb, g_err_wb = validate_generator(generator, recognizer, char_vector, valid2_dataset, batch_size, latent_dim,
                                            valid2_words, wbs)

    return g_err_vb, g_err_wb, r_err_vb, r_err_wb


def prepare_gan_input(batch_size, latent_dim, random_words, bucket_size, buckets, bucket_position):
    # prepare Gen. input
    noise = tf.random.normal([batch_size, latent_dim])
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

    return noise, fake_labels


def train(train_dataset, valid1_dataset, valid2_dataset, generator, discriminator, recognizer, composite_gan,
          checkpoint, generator_optimizer, discriminator_optimizer, recognizer_optimizer, seed_labels, batch_size,
          epochs, latent_dim, gen_path, loss_fn, disc_iters, apply_gradient_balance, gradient_balance_type,
          random_words, bucket_size, char_vector, train_words, valid1_words, valid2_words):
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

    # best_char_error_rate = float('inf')  # best validation character error rate
    # no_improvement_since = 0  # number of epochs no improvement of character error rate occurred

    # define checkpoint save paths
    generator_save_dir = os.path.join(checkpoint, 'generator/')
    recognizer_save_dir = os.path.join(checkpoint, 'recognizer/')

    # create folders to save models
    if not os.path.exists(generator_save_dir):
        os.makedirs(generator_save_dir)
    if not os.path.exists(recognizer_save_dir):
        os.makedirs(recognizer_save_dir)

    # create summary files
    batch_summary = open(gen_path + "batch_summary.txt", "w")
    epoch_summary = open(gen_path + "epoch_summary.txt", "w")

    # write headers to files
    header = "disc_loss;disc_loss_real;disc_loss_fake;r_loss_real;r_loss_fake;r_loss_balanced;g_loss;g_lossT;g_lossS;" \
             "g_loss_final;alpha;r_loss_fake_std;g_loss_std"
    epoch_summary.write(header + "cer_g_wb;cer_g_vb;cer_r_wb;cer_r_vb\n")
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
    for epoch_idx in range(epochs):
        start = time.time()

        # variables for total of losses
        d_loss_total = 0.0
        d_loss_real_total = 0.0
        d_loss_fake_total = 0.0
        r_loss_fake_total = 0.0
        r_loss_real_total = 0.0
        r_loss_balanced_total = 0.0
        g_loss_total = 0.0
        g_loss_added_total = 0.0
        g_loss_balanced_total = 0.0
        g_loss_final_total = 0.0
        g_loss_std_total = 0.0
        r_loss_fake_std_total = 0.0
        alphas = 0.0

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
            noise, fake_labels = prepare_gan_input(batch_size, latent_dim, random_words, bucket_size, buckets,
                                                   bucket_position)

            # if we ran out of a bucket in random words, choose another
            if fake_labels is None:
                continue

            # training step
            r_loss_fake, r_loss_real, r_loss_balanced, g_loss, g_loss_added, g_loss_balanced, d_loss, d_loss_real, \
            d_loss_fake, g_loss_final, alpha, r_loss_fake_std, g_loss_std = train_step(epoch_idx, batch_idx,
                                                                                       batch_per_epoch, image_batch,
                                                                                       label_batch, discriminator,
                                                                                       recognizer, composite_gan,
                                                                                       generator_optimizer,
                                                                                       discriminator_optimizer,
                                                                                       recognizer_optimizer, batch_size,
                                                                                       loss_fn, disc_iters,
                                                                                       apply_gradient_balance,
                                                                                       gradient_balance_type,
                                                                                       noise, fake_labels)

            # write batch summary to file
            batch_summary.write(str(d_loss) + ";" + str(d_loss_real) + ";" + str(d_loss_fake) + ";" +
                                str(r_loss_real) + ";" + str(r_loss_fake) + ";" + str(r_loss_balanced) + ";" +
                                str(g_loss) + ";" + str(g_loss_added) + ";" + str(g_loss_balanced) + ";" +
                                str(g_loss_final) + ";" + str(alpha) + ";" + str(r_loss_fake_std) + ";" +
                                str(g_loss_std) + '\n')

            # append to lists for epoch summary
            d_loss_total += d_loss
            d_loss_real_total += d_loss_real
            d_loss_fake_total += d_loss_fake
            r_loss_fake_total += r_loss_fake
            r_loss_real_total += r_loss_real
            r_loss_balanced_total += r_loss_balanced
            g_loss_total += g_loss
            g_loss_added_total += g_loss_added
            g_loss_balanced_total += g_loss_balanced
            g_loss_final_total += g_loss_final
            g_loss_std_total += g_loss_std
            r_loss_fake_std_total += r_loss_fake_std
            alphas += alpha

        # validate Generator and Recognizer with CER
        g_err_wb, g_err_vb, r_err_wb, r_err_vb = validate(generator, recognizer, char_vector, valid1_dataset,
                                                          valid2_dataset, batch_size, latent_dim, valid1_words,
                                                          valid2_words, train_words)

        divider = batch_per_epoch
        epoch_summary.write(str(d_loss_total / divider) + ";" + str(d_loss_real_total / divider) + ";" +
                            str(d_loss_fake_total / divider) + ";" + str(r_loss_real_total / divider) + ";" +
                            str(r_loss_fake_total / divider) + ";" + str(r_loss_balanced_total / divider) + ";" +
                            str(g_loss_total / divider) + ";" + str(g_loss_added_total / divider) + ";" +
                            str(g_loss_balanced_total / divider) + ";" + str(g_loss_final_total / divider) + ";" +
                            str(alphas / divider) + ";" + str(r_loss_fake_std_total / divider) + ";" +
                            str(g_loss_std_total / divider) + ";" + str(g_err_wb) + ";" + str(g_err_vb) + ";" +
                            str(r_err_wb) + ";" + str(r_err_vb) + ";" + '\n')

        # produce images for visual evaluation - quantitative
        generate_and_save_images(generator, epoch_idx + 1, seed_labels, gen_path, char_vector)

        # define sub-folders to save weights after each epoch
        generator_epoch_save = os.path.join(generator_save_dir, str(epoch_idx + 1) + '/')
        recognizer_epoch_save = os.path.join(recognizer_save_dir, str(epoch_idx + 1) + '/')

        # create folders to save models
        if not os.path.exists(generator_epoch_save):
            os.makedirs(generator_epoch_save)
        if not os.path.exists(recognizer_epoch_save):
            os.makedirs(recognizer_epoch_save)

        # save generator
        generator.save_weights(generator_epoch_save + 'cktp-' + str(epoch_idx + 1))
        # save recognizer
        recognizer.save_weights(recognizer_epoch_save + 'cktp-' + str(epoch_idx + 1))

        print('Time for epoch {} is {} sec'.format(epoch_idx + 1, time.time() - start))

        for sublist in random_words:
            random.shuffle(sublist)

        #     # if best validation accuracy so far, save model parameters
        # if char_err < best_char_error_rate:
        #     print('Character error rate improved, save model')
        #     best_char_error_rate = char_err
        #     no_improvement_since = 0
        # else:
        #     print(f'Character error rate not improved, best so far: {char_err}%')
        #     no_improvement_since += 1
        #
        # # stop training if no more improvement in the last x epochs
        # if no_improvement_since >= 15:
        #     print(f'No more improvement since {15} epochs. Training stopped.')
        #     break

    batch_summary.close()
    epoch_summary.close()


def train_step(epoch_idx, batch_idx, batch_per_epoch, images, labels, discriminator, recognizer, composite_gan,
               generator_optimizer, discriminator_optimizer, recognizer_optimizer, batch_size, loss_fn, disc_iters,
               apply_gradient_balance, gradient_balance_type, noise, fake_labels):
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
    print(fake_labels)
    print(fake_labels.dtype)
    print(fake_labels[0])
    print(fake_labels[0].dtype)
    print(len(fake_labels[0]))
    sequence_length_fake = len(fake_labels[0])

    # compute loss & update gradients
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as rec_tape:
        # generate images + compute D(fake) + R(fake)
        inp_len_fake = -1 + sequence_length_fake * 4

        gen_images, d_fake_logits, r_fake_logits = composite_gan(
            [noise, fake_labels, np.array([[inp_len_fake]] * batch_size),
             np.array([[sequence_length_fake]] * batch_size)], training=True)

        # compute D(real)
        d_real_logits = discriminator([images], training=True)

        # compute R(real)
        inp_len_real = -1 + sequence_length_real * 4
        time_steps = recognizer(images, training=True)

        r_real_logits = ctc_loss(labels, time_steps, np.array([[inp_len_real]] * batch_size_real),
                                 np.array([[sequence_length_real]] * batch_size_real))

        # compute losses
        d_loss, d_loss_real, d_loss_fake, g_loss = loss_fn(d_real_logits, d_fake_logits)

        # apply gradient balancing (optional)
        g_loss_balanced, r_loss_balanced, alpha, r_loss_fake_std, g_loss_std = apply_gradient_balancing(r_fake_logits,
                                                                                                        g_loss, gradient_balance_type,
                                                                                                        alpha=1)
        g_loss_added = g_loss + r_fake_logits
        if apply_gradient_balance:
            g_loss_final = g_loss_balanced
        else:
            g_loss_final = g_loss_added

        # compute stats
        r_loss_fake_mean = tf.reduce_mean(r_fake_logits)
        r_loss_real_mean = tf.reduce_mean(r_real_logits)
        r_loss_balanced_mean = tf.reduce_mean(r_loss_balanced)
        g_loss_mean = tf.reduce_mean(g_loss)
        g_loss_added_mean = tf.reduce_mean(g_loss_added)
        g_loss_balanced_mean = tf.reduce_mean(g_loss_balanced)
        d_loss_mean = tf.reduce_mean(d_loss)
        d_loss_real_mean = tf.reduce_mean(d_loss_real)
        d_loss_fake_mean = tf.reduce_mean(d_loss_fake)
        g_loss_final_mean = tf.reduce_mean(g_loss_final)

    tf.print('>%d, %d/%d, d=%.3f, d_real=%.3f, d_fake=%.3f, g_trad=%.3f, r_loss_fake=%.3f, g_loss=%.3f, r=%.3f' % (
        epoch_idx + 1, batch_idx + 1, batch_per_epoch, d_loss_mean, d_loss_real_mean, d_loss_fake_mean, g_loss_mean,
        r_loss_fake_mean, g_loss_final_mean, r_loss_real_mean))

    # compute and apply gradients of D and R
    discriminator.trainable = True
    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    recognizer.trainable = True
    gradients_of_recognizer = rec_tape.gradient(r_real_logits, recognizer.trainable_variables)
    recognizer_optimizer.apply_gradients(zip(gradients_of_recognizer, recognizer.trainable_variables))

    # take disc_iters (default 1) D steps per G step
    if (batch_idx + 1) % disc_iters == 0:
        # compute and apply gradients of G
        recognizer.trainable = False
        discriminator.trainable = False
        gradients_of_generator = gen_tape.gradient(g_loss_final, composite_gan.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, composite_gan.trainable_variables))

    return r_loss_fake_mean.numpy(), r_loss_real_mean.numpy(), r_loss_balanced_mean.numpy(), g_loss_mean.numpy(), \
           g_loss_added_mean.numpy(), g_loss_balanced_mean.numpy(), d_loss_mean.numpy(), d_loss_real_mean.numpy(), \
           d_loss_fake_mean.numpy(), g_loss_final_mean.numpy(), alpha, r_loss_fake_std.numpy(), g_loss_std.numpy()


def apply_gradient_balancing(r_fake_logits, g_loss, gradient_balance_type, alpha=1):
    """
    gradient balancing based on https://arxiv.org/pdf/2003.10557.pdf

    :param r_fake_logits:   output of tf.keras.backend.ctc_batch_cost
    :param g_loss:          output of tf.nn.sigmoid_cross_entropy_with_logits
    :param alpha:           controls relative importance of L_R and L_D
    :return:
    """
    # g_loss = - disc_fake_logits
    r_loss_fake_std = tf.math.reduce_std(r_fake_logits)
    g_loss_std = tf.math.reduce_std(g_loss)
    if gradient_balance_type == 1:
        r_loss_balanced = alpha * (((g_loss_std) / (r_loss_fake_std)) * r_fake_logits)
    else:
        r_loss_fake_mean = tf.math.reduce_mean(r_fake_logits)
        g_loss_mean = tf.math.reduce_mean(g_loss)
        r_loss_balanced = alpha * (((g_loss_std / r_loss_fake_std) * (r_fake_logits-r_loss_fake_mean)) - g_loss_mean)
    g_balanced = g_loss + r_loss_balanced
    return g_balanced, r_loss_balanced, alpha, r_loss_fake_std, g_loss_std


def generate_and_save_images(model, epoch, test_input, gen_path, char_vector):
    """
    Generate and save predictions + ground truth label

    :param model:
    :param epoch:
    :param test_input:
    :param gen_path:
    :param char_vector:
    :return:
    """

    seeds, labels = test_input
    predictions = []
    for i in range(len(seeds)):
        # get a seed/style
        seed = seeds[i]
        # generate all labels for the style
        for j in range(len(labels)):
            label = tf.convert_to_tensor(labels[j])
            label = tf.expand_dims(label, axis=0)
            # generate image with the seed and label
            generated_imgs = model([seed, label], training=False)
            # scaling change: (-1, 1) -> (0, 1)
            generated_imgs = (generated_imgs + 1) / 2.0
            if np.isnan(generated_imgs).any():
                predictions.append(np.ones_like(generated_imgs))  # if image is nan, paint black image
            else:
                predictions.append(generated_imgs)

    # grid to scale images with different width
    grid = [pred.shape[-2] // 16 for pred in predictions]
    grid = grid[:len(labels)]

    sub_plot_x = len(seeds)
    sub_plot_y = len(labels)
    f, axs = plt.subplots(sub_plot_x, sub_plot_y, gridspec_kw={'width_ratios': grid}, dpi=300)
    for i in range(len(seeds)):
        for j in range(len(labels)):
            # write images
            axs[i, j].imshow(predictions[(i * len(labels)) + j][0, :, :, 0], cmap='gray')
            # write labels once
            if i == 0:
                axs[i, j].text(-5, -10, "".join([char_vector[label] for label in labels[j]]), fontsize='xx-small')
            axs[i, j].axis('off')

    if not os.path.exists(gen_path):
        os.makedirs(gen_path)

    plt.savefig(os.path.join(gen_path, 'image_at_epoch_{:04d}.png'.format(epoch)))


def make_gif(gen_path):
    """
    Use imageio to create an animated gif using the images saved during training; based on
    https://www.tensorflow.org/tutorials/generative/dcgan

    :param gen_path:
    :return:
    """
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    anim_file = gen_path + 'biggan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(gen_path + 'image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def load_random_word_list(reading_dir, bucket_size, char_vector):
    """
    Helper to produce random word list; encode each word into vector defined by char_vector
    e.g. 'auto' -> [0, 20, 19, 14]

    :param reading_dir:         input dir of random_words.txt
    :param bucket_size:         max transcription length of random word
    :param char_vector:         index of char within charvector represents char encoding
    :return:
    """

    random_words = []
    for i in range(bucket_size):
        random_words.append([])

    random_words_path = os.path.dirname(os.path.dirname(reading_dir)) + '/'
    with open(os.path.join(random_words_path, 'brown_random.txt'), 'r') as fi_random_word_list:
        for word in fi_random_word_list:
            word = word.strip()
            bucket = len(word)

            if bucket <= bucket_size and word.isalpha():
                random_words[bucket - 1].append([char_vector.index(char) for char in word])

    return random_words
