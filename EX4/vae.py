from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_1 = layers.Dense(intermediate_dim, activation="relu")
        self.dense_2 = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_1 = layers.Dense(intermediate_dim, activation="relu")
        self.dense_2 = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="relu")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.dense_output(x)


class VariationalAutoEncoder(Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=256,
        latent_dim=32,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

if __name__ == "__main__":
    original_dim = 784
    # vae = VariationalAutoEncoder(original_dim, 64, 32)
    #
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # mse_loss_fn = tf.keras.losses.MeanSquaredError()
    #
    # loss_metric = tf.keras.metrics.Mean()

    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255

    vae = VariationalAutoEncoder(784, 256, 2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    vae.fit(x_train, x_train, epochs=20, batch_size=128)
    prova = vae(np.expand_dims(x_train[1], axis=0))
    prova = tf.reshape(prova, [28, 28])
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(tf.reshape(x_train[1], [28, 28]))
    axs[1].imshow(prova)
    plt.show()


    # epochs = 2
    #
    # # Iterate over epochs.
    # for epoch in range(epochs):
    #     print("Start of epoch %d" % (epoch,))
    #
    #     # Iterate over the batches of the dataset.
    #     for step, x_batch_train in enumerate(train_dataset):
    #         with tf.GradientTape() as tape:
    #             reconstructed = vae(x_batch_train)
    #             # Compute reconstruction loss
    #             loss = mse_loss_fn(x_batch_train, reconstructed)
    #             loss += sum(vae.losses)  # Add KLD regularization loss
    #
    #         grads = tape.gradient(loss, vae.trainable_weights)
    #         optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    #
    #         loss_metric(loss)
    #
    #         if step % 100 == 0:
    #             print("step %d: mean loss = %.4f" % (step, loss_metric.result()))