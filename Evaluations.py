import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Definining the discriminator loss

def discriminator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

# Defining the generator loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def calc_accuracy(prediction):
  """
    Function that takes in the some data judgements
    from the discriminator and get the average of
    judgements that indicate how the discriminator is fooled.
  """
  prediction_clipped = tf.clip_by_value(prediction, 0.0, 1.0, name=None)
  return tf.reduce_mean(prediction_clipped)