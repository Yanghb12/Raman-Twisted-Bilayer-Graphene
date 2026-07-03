import tensorflow as tf
from Models import make_generator_model as generator
from Models import make_discriminator_model as discriminator
from Evaluations import generator_loss, discriminator_loss, calc_accuracy
import time

BATCH_SIZE = 4
noise_dim = 113

generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

def train_step(data):
    """
      Function for implementing one training step
      of the GAN model
    """
    noise = tf.random.normal([BATCH_SIZE, noise_dim], seed=1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)

        real_output = discriminator(data, training=True)
        fake_output = discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        acc = calc_accuracy(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss, acc


def train(dataset, epochs):
    """
      Main GAN Training Function
    """
    epochs_gen_losses, epochs_disc_losses, epochs_accuracies = [], [], []

    for epoch in range(epochs):
        start = time.time()

        gen_losses, disc_losses, accuracies = [], [], []

        for data_batch in dataset:
            gen_loss, disc_loss, acc = train_step(data_batch)
            accuracies.append(acc)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        epoch_gen_loss = np.average(gen_losses)
        epoch_disc_loss = np.average(disc_losses)
        epoch_accuracy = np.average(accuracies)
        epochs_gen_losses.append(epoch_gen_loss)
        epochs_disc_losses.append(epoch_disc_loss)
        epochs_accuracies.append(epoch_accuracy)
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        print("Generator Loss: {}, Discriminator Loss: {}".format(epoch_gen_loss, epoch_disc_loss))
        print("Accuracy: {}".format(epoch_accuracy))

        # Draw the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            draw_training_evolution(generator, epoch + 1)

        # Save the model every 2 epochs for the last 2000 epochs
        if (epoch + 1) % 2 == 0 and epoch > (numofEPOCHS - 2000):
            checkpoint.save(file_prefix=checkpoint_prefix)  # Comment not to save model checkpoints while training

    return epochs_gen_losses, epochs_disc_losses, epochs_accuracies