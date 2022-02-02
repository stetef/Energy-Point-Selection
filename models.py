"""Wasserstein GAN."""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K

from keras.constraints import Constraint
from keras.layers import Dense, BatchNormalization, LeakyReLU
from keras.models import Sequential
from keras.optimizers import RMSprop

from utils import generate_linear_combos, get_coeffs_from_sepctra
from utils import loss, scale_coeffs_to_add_to_one


class ClipConstraint(Constraint):
    """Custom weight clipping object."""

    def __init__(self, clip_value):
        """init."""
        self.clip_value = clip_value

    def __call__(self, weights):
        """call."""
        return tf.clip_by_value(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        """config."""
        return {'clip_value': self.clip_value}


class WGAN:
    """WGAN."""

    def __init__(self, refs, latent_dim):
        """Init function."""
        self.Refs = refs
        self.y_shape = self.Refs.shape[1]
        self.latent_dim = latent_dim
        self.discriminator = self._define_discriminator(n_inputs=self.y_shape)
        self.generator = self._define_generator(n_outputs=self.y_shape)
        self.gan = self._define_gan(self.generator, self.discriminator)

    def _discriminator_loss(self, real_img, fake_img):
        """Discriminator loss."""
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    def _wasserstein_loss(self, y_true, y_pred):
        """GAN loss for Wasserstein distance."""
        return K.mean(y_true * y_pred)

    def _define_discriminator(self, n_inputs=2):
        """Initialize discriminator."""
        constraint = ClipConstraint(0.01)
        model = Sequential(name='Discriminator')
        model.add(Dense(32, kernel_initializer='he_uniform',
                        input_dim=n_inputs, kernel_constraint=constraint))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(1, activation='linear'))
        optimizer = RMSprop(lr=0.00005)
        model.compile(loss=self._discriminator_loss, optimizer=optimizer)
        return model

    def _define_generator(self, n_outputs=2):
        """Initialize generator."""
        model = Sequential(name='Generator')
        model.add(Dense(32, kernel_initializer='he_uniform',
                        input_dim=self.latent_dim))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(n_outputs, activation='tanh'))
        return model

    def _generate_latent_points(self, n):
        """Create noise for generator input."""
        return np.random.normal(loc=0., scale=0.5, size=(n, self.latent_dim))

    def generate_fake_samples(self, n, training=True):
        """Pass through generator."""
        noise = self._generate_latent_points(n)
        spectra = self.generator.predict(noise)
        spectra = (spectra + 1) / 2.0
        if training:
            return spectra, np.ones((n, 1))
        else:
            return spectra, get_coeffs_from_sepctra(spectra, self.Refs)

    def _define_gan(self, generator, critic):
        """Create WGAN."""
        for layer in critic.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        model = Sequential(name='GAN')
        model.add(generator)
        model.add(critic)
        optimizer = RMSprop(lr=0.00005)
        model.compile(loss=self._wasserstein_loss, optimizer=optimizer)
        return model

    def plot_history(self, real_hist, fake_hist, g_hist):
        plt.plot(real_hist, label='real critic loss')
        plt.plot(fake_hist, label='gen critic loss')
        plt.plot(g_hist, label='gan loss')
        plt.legend()
        plt.show()

    def train_critic(self, half_batch, losses, **kwargs):
        """Train the discriminator."""
        X_real, y_real = generate_linear_combos(self.Refs, **kwargs)
        c_loss1 = self.discriminator.train_on_batch(X_real, y_real)
        X_fake, y_fake = self.generate_fake_samples(half_batch,
                                                     training=True)
        c_loss2 = self.discriminator.train_on_batch(X_fake, y_fake)
        losses[0].append(c_loss1)
        losses[1].append(c_loss2)
        return losses

    def train_gan(self, n_batch):
        """GAN training step."""
        X_gan = self._generate_latent_points(n_batch)
        y_gan = -np.ones((n_batch, 1))
        g_loss = self.gan.train_on_batch(X_gan, y_gan)
        return g_loss

    def train(self, n_epochs=128, n_batch=128, n_critic=25,
    	      verbose=True, plot=True):
        """Training function."""
        half_batch = int(n_batch / 2)
        kwargs = {'N': half_batch, 'scale': 0.0, 'dropout': 0.1 ,
                  'training': True}
        critic_real_loss, critic_fake_loss, gan_loss = [], [], []
        for i in range(n_epochs):
            critic_losses = [[], []]
            for j in range(n_critic):
                critic_losses = self.train_critic(half_batch, critic_losses,
                                                  **kwargs)
            critic_real_loss.append(-np.mean(critic_losses[0]))
            critic_fake_loss.append(np.mean(critic_losses[1]))
            g_loss = self.train_gan(n_batch)
            gan_loss.append(-g_loss)
            if verbose:
                print(f'Epoch: {i + 1}', end='\r')
        if plot:
            self.plot_history(critic_real_loss, critic_fake_loss, gan_loss)
