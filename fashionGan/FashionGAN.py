import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime
import time

# globals:
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
GEN_LR = 5e-5
DISC_LR = 3e-5
EPOCHS = 150
BATCH_SIZE = 256
BUFFER_SIZE = 60000
NOISE_DIMS = 100
NUM_EXAMPLES_TO_GENERATE = 16
SEED = 42

noise = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIMS])


# Generator
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(7 * 7 * 256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# Discriminator:
def make_discriminator_model():

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding="same", input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(1))

    return model


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_accuracy(fake_output):
    return tf.math.reduce_sum(tf.where(fake_output > 0.5, 1, 0)) / fake_output.shape[0]


def discriminator_accuracy(real_output, fake_output):
    negatives = tf.math.reduce_sum(tf.where(fake_output > 0.5, 0, 1))
    positives = tf.math.reduce_sum(tf.where(real_output > 0.5, 1, 0))
    return (negatives + positives) / (real_output.shape[0] + fake_output.shape[0])


def generate_image(generator):

    predictions = generator(noise, training=False)

    plt.figure(figsize=(4, 4))

    for i in range(NUM_EXAMPLES_TO_GENERATE):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :, 0] + 127.5) * 127.5, cmap='gray')
        plt.axis('off')

    plt.show()


if __name__ == '__main__':

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    graph_writer = tf.summary.create_file_writer(log_dir + "/graphs")

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, _), (_, _) = fashion_mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model()
    discriminator = make_discriminator_model()


    @tf.function
    def train_step(images, epoch):

        noise_ = tf.random.normal([BATCH_SIZE, NOISE_DIMS])

        if epoch == 15:
            GEN_LR = 2e-5
            DISC_LR = 1e-5

        if epoch == 50:
            GEN_LR = 8e-6
            DISC_LR = 8e-6

        gen_optimizer = tf.keras.optimizers.Adam(GEN_LR)
        disc_optimizer = tf.keras.optimizers.Adam(DISC_LR)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise_, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        if np.random.random() > 0.1:
            disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        else:
            gradients_of_generator = 0

        with file_writer.as_default():
            tf.summary.scalar("generator_accuracy", generator_accuracy(fake_output), step=gen_optimizer.iterations)
            tf.summary.scalar("discriminator_accuracy", discriminator_accuracy(real_output, fake_output),
                              step=gen_optimizer.iterations)
            tf.summary.scalar("generator_loss", gen_loss, step=gen_optimizer.iterations)
            tf.summary.scalar("discriminator_loss", disc_loss, step=gen_optimizer.iterations)


    def train(dataset, epochs):
        generate_image(generator)
        for epoch in range(epochs):
            start = time.time()

            for batch in dataset:

                train_step(batch, epoch)

            print("Epoch {} finished. Training time: {}".format(epoch, time.time() - start))
            generate_image(generator)

        generate_image(generator)


    tf.summary.trace_on(graph=True)
    train(dataset, EPOCHS)
    with graph_writer.as_default():
        tf.summary.trace_export(name="graph", step=0)
    generator.save("models\FashionGenerator.h5")
    discriminator.save("models\FashionDiscriminator.h5")