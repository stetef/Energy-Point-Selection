"""GAN models."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K

from keras.constraints import Constraint
from keras.layers import Dense, BatchNormalization, LeakyReLU
from keras.models import Sequential
from keras.optimizers import RMSprop

from utils import generate_linear_combos, get_coeffs_from_sepctra
from utils import format_axis


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
    """
    WGAN.

    A Wasserstein GAN uses the Wasserstein/ Earth mover distance as a loss,
    weight clipping, and a 'critic' instead of a discriminator
    to help with training and convergence.
    """

    def __init__(self, refs, latent_dim):
        """Init function."""
        self.Refs = refs
        self.y_shape = self.Refs.shape[1]
        self.latent_dim = latent_dim
        self.discriminator = self._define_discriminator(n_inputs=self.y_shape)
        self.generator = self._define_generator(n_outputs=self.y_shape)
        self.gan = self._define_gan(self.generator, self.discriminator)

    def _discriminator_loss(self, real_data, fake_data):
        """Discriminator loss."""
        real_loss = tf.reduce_mean(real_data)
        fake_loss = tf.reduce_mean(fake_data)
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
        optimizer = keras.optimizers.Adam(learning_rate=0.0002,
                                          beta_1=0.5, beta_2=0.9)
        model.compile(loss=self._discriminator_loss, optimizer=optimizer)
        return model

    def _define_generator(self, n_outputs=2):
        """Initialize generator."""
        model = Sequential(name='Generator')
        model.add(Dense(32, kernel_initializer='he_uniform',
                        input_dim=self.latent_dim))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(32, kernel_initializer='he_uniform',
                        input_dim=32))
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
        # spectra = (spectra + 1.0) / 2.0
        if training:
            return spectra.astype(np.float64), np.ones((n, 1))
        else:
            return spectra.astype(np.float64), get_coeffs_from_sepctra(spectra,
                                                                       self.Refs)

    def _define_gan(self, generator, critic):
        """Create WGAN."""
        for layer in critic.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        model = Sequential(name='GAN')
        model.add(generator)
        model.add(critic)
        optimizer = keras.optimizers.Adam(learning_rate=0.00001,
                                          beta_1=0.5, beta_2=0.9)
        model.compile(loss=self._wasserstein_loss, optimizer=optimizer)
        return model

    def plot_history(self, critic_losses, gan_losses):
        """Plot showing training loss."""
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot(critic_losses, label='critic loss')
        # plt.plot(gan_losses, label='gan loss')
        xticks = np.array(ax.get_xticks(), dtype=int)
        ax.set_xticklabels(xticks, fontsize=16)
        ax.set_yticklabels(np.array(ax.get_yticks(), dtype=float), fontsize=16)
        ax.set_xlabel('Epoch', fontsize=18)
        ax.set_ylabel('Loss', fontsize=18)
        ax.tick_params(direction='in', width=2, length=8, which='major')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.legend(fontsize=18, loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.show()

    def plot_input_output(self, m, Energy, Refs):
        """GAN visualization."""
        latent_points = self._generate_latent_points(m)
        spectra = self.generator.predict(latent_points)
        spectra = (spectra + 1) / 2.0

        fig, axes = plt.subplots(figsize=(5 * m, 5), ncols=m)
        plt.subplots_adjust(wspace=0)
        for i in range(m):
            ax = axes[i]
            n = len(latent_points[i])
            ax.plot(Energy, spectra[i], '-', linewidth=4, c=plt.cm.tab20(0),
                    label='output')
            ax.plot(np.linspace(min(Energy), max(Energy), n),
                    latent_points[i], '-', linewidth=4, c=plt.cm.tab20(2),
                    label='input')
            format_axis(ax, ticks=(10, 20), fontsize=20)
            ax.legend(fontsize=20, loc=4)
        plt.show()

    def train_critic(self, half_batch, losses, **kwargs):
        """Train the discriminator."""
        X_real, y_real = generate_linear_combos(self.Refs, **kwargs)
        self.discriminator.train_on_batch(X_real, y_real)
        X_fake, y_fake = self.generate_fake_samples(half_batch,
                                                    training=True)
        self.discriminator.train_on_batch(X_fake, y_fake)
        critic_loss = self._discriminator_loss(real_data=X_real,
                                               fake_data=X_fake)
        losses.append(critic_loss)
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
        kwargs = {'N': half_batch, 'scale': 0.0, 'dropout': 0.1,
                  'training': True}
        critic_losses, gan_losses = [], []
        for i in range(n_epochs):
            critic_loss = []
            for j in range(n_critic):
                critic_loss = self.train_critic(half_batch, critic_loss,
                                                **kwargs)
            critic_losses.append(np.average(critic_loss))
            gan_loss = self.train_gan(n_batch)
            gan_losses.append(-gan_loss)
            if verbose:
                print(f'Epoch: {i + 1}', end='\r')
        if plot:
            self.plot_history(critic_losses, gan_losses)


class WGAN_GP(WGAN):
    """
    WGAN-GP.

    A WGAN-GP uses gradient penalty instead of the weight clipping
    to enforce the Lipschitz constraint.
    """

    def __init__(self, gp_weight, *args):
        """
        Init function.

        Attributes:
            gp_weight - extra parameter for the weight of the graident penalty

        Inherit all other attributes from a normal WGAN.
        """
        self.gp_weight = gp_weight
        super().__init__(*args)

    def _define_discriminator(self, n_inputs=2):
        """
        Initialize discriminator.

        The critic for a WGAN-GP does not have weight clipping 
        or batch normalization.
        """
        model = Sequential(name='Discriminator')
        model.add(Dense(32, kernel_initializer='he_uniform',
                        input_dim=n_inputs))
        model.add(LeakyReLU())
        model.add(Dense(1, activation='linear'))
        optimizer = keras.optimizers.Adam(learning_rate=0.0001,
                                          beta_1=0.5, beta_2=0.9)
        model.compile(loss=self._discriminator_loss, optimizer=optimizer)
        return model

    def gradient_penalty(self, batch_size, real_data, fake_data):
        """
        Calculate the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Start with the interpolated data
        alpha = tf.random.normal([batch_size, self.y_shape], 0.0, 1.0)
        # diff = fake_data - real_data
        # interpolated = real_data + alpha * diff
        interpolated = alpha * real_data + (1 - alpha)*fake_data

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated data.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated data.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 0.0) ** 2)
        return tf.cast(gp, tf.float64)

    def train_critic(self, half_batch, losses, **kwargs):
        """Train the discriminator."""
        X_real, y_real = generate_linear_combos(self.Refs, **kwargs)
        self.discriminator.train_on_batch(X_real, y_real)
        X_fake, y_fake = self.generate_fake_samples(half_batch,
                                                    training=True)
        self.discriminator.train_on_batch(X_fake, y_fake)
        temp_cost = self._discriminator_loss(real_data=X_real,
                                             fake_data=X_fake)
        gp = self.gradient_penalty(half_batch, X_real, X_fake)
        critic_loss = temp_cost + gp * self.gp_weight
        losses.append(critic_loss)
        with tf.GradientTape() as tape:
            d_gradient = tape.gradient(critic_loss,
                                       self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        return losses
