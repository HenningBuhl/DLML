from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

'''
MNIST GAN example from tensorflow https://www.tensorflow.org/tutorials/generative/dcgan
'''

# global variables:

# Cross-entropy-Loss for binary classification
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Training hyperparams
EPOCHS = 50
GEN_LR = 1e-4       # generator learning rate
DISC_LR = 1e-4      # discriminator learning rate
noise_dim = 100     # seed dimension
num_examples_to_generate = 16   # during the training process we show some example generated images, this is how many

# Safe the seed for the examples to visualize progress during training
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# THE GENERATOR:
def make_generator_model():
    model = tf.keras.Sequential()
    # So it takes a 100-dimensional Random vector as input:
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # None is the batch size

    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding="same", use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding="same", use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# THE DISCRIMINATOR:
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


''' 
Compute the Loss for the discriminator by comparing the predictions on real images to an array of 1s, and the
predictions on fake (generated) images to an array of 0s 
'''
def discriminator_loss(real_output, fake_output):
    # The output for all real images should be 1
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # The output for all fake images should be 0
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss
    return total_loss


'''
Compute the loss for the generator. The generator tries to maximize the discriminators output for the generated
images. 
'''
def generator_loss(fake_output):
    # If the output is 1 for a fake/generated image, reward
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('generated_images' + os.sep + 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


if __name__ == '__main__':
    # load the mnist data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # reshape and normalize the images
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Instantiate the two models
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # Optimizers for the two models
    generator_optimizer = tf.keras.optimizers.Adam(GEN_LR)
    discriminator_optimizer = tf.keras.optimizers.Adam(DISC_LR)

    # Custom training loop
    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate the fake images from random noise
            generated_images = generator(noise, training=True)

            # Forward real and fake images through the discriminator to generate the predictions
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            # Use the Adversarial Loss to train both models:
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def train(dataset, epochs):
        print("Starting training...")
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                train_step(image_batch)

            # Produce the images for visualisation of training
            generate_and_save_images(generator, epoch + 1, seed)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Final images:
        generate_and_save_images(generator, epochs, seed)

    train(train_dataset, EPOCHS)