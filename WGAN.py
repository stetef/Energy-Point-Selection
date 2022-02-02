"""Wasserstein GAN."""

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

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

    def __init__(self, Refs, latent_dim):
        """Init function."""
        self.Refs = Refs
        self.y_shape = Refs.shape[1]
        self.latent_dim = latent_dim
        self.discriminator = self._define_discriminator(n_inputs=self.y_shape)
        self.generator = self._define_generator(self.latent_dim,
                                                n_outputs=self.y_shape)
        self.gan = self._define_gan(self.generator, self.discriminator)


    def _discriminator_loss(self, real_img, fake_img):
        """Discriminator loss."""
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    def _wasserstein_loss(self, y_true, y_pred):
        """GAN loss for Wasserstein distance."""
        return tf.mean(y_true * y_pred)

    def _define_discriminator(self, n_inputs=2):
        """Initialize discriminator."""
        constraint = ClipConstraint(0.01)
        model = Sequential(name='Discriminator')
        model.add(Dense(32, kernel_initializer='he_uniform', input_dim=n_inputs,
                        kernel_constraint=constraint))
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
            return spectra, get_coeffs_from_sepctra(spectra)

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
