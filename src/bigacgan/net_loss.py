import tensorflow as tf


def not_saturating(d_real_logits, d_fake_logits, s_styleimgs_logits, s_trainingimgs_logits, s_fake_logits):
    """
    Returns the discriminator and generator loss for Non-saturating loss; based on
    https://github.com/google/compare_gan/blob/master/compare_gan/gans/loss_lib.py

    :param d_real_logits:
    :param d_fake_logits:
    :return:
    """

    d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits),
                                                          name="cross_entropy_d_real")
    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits),
                                                          name="cross_entropy_d_fake")
    d_loss = d_loss_real + d_loss_fake

    s_styleimgs_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=s_styleimgs_logits,
                                                                 labels=tf.ones_like(s_styleimgs_logits),
                                                                 name="cross_entropy_s_style")
    s_iam_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=s_trainingimgs_logits,
                                                               labels=tf.zeros_like(s_trainingimgs_logits),
                                                               name="cross_entropy_s_iam")
    s_loss = s_styleimgs_loss + s_iam_loss

    g_loss_disc = tf.nn.sigmoid_cross_entropy_with_logits(logits=(d_fake_logits), labels=tf.ones_like(d_fake_logits),
                                                          name="cross_entropy_g_disc")
    g_loss_style = tf.nn.sigmoid_cross_entropy_with_logits(logits=(s_fake_logits), labels=tf.ones_like(s_fake_logits),
                                                          name="cross_entropy_g_style")

    g_loss = g_loss_disc + g_loss_style

    return d_loss, d_loss_real, d_loss_fake, g_loss, s_loss, s_styleimgs_loss, s_iam_loss


def hinge(d_real_logits, d_fake_logits, s_real_logits, s_fake_logits):
    """
    Returns the discriminator and generator loss for the hinge loss; based on
    https://github.com/google/compare_gan/blob/master/compare_gan/gans/loss_lib.py

    :param d_real_logits: logits for real points, shape [batch_size, 1].
    :param d_fake_logits: logits for fake points, shape [batch_size, 1].
    :return:
    """
    d_loss_real = tf.nn.relu(1.0 - d_real_logits)
    d_loss_fake = tf.nn.relu(1.0 + d_fake_logits)
    d_loss = d_loss_real + d_loss_fake
    s_loss_real = tf.nn.relu(1.0 - s_real_logits)
    s_loss_fake = tf.nn.relu(1.0 + s_fake_logits)
    s_loss = s_loss_real + s_loss_fake
    g_loss = - (d_fake_logits + s_fake_logits)
    return d_loss, d_loss_real, d_loss_fake, g_loss, s_loss, s_loss_real, s_loss_fake
