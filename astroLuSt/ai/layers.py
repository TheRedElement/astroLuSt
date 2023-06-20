
#%%imports
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


#%%definitions
class Sampling(Layer):
    """
        - sampling layer
        - uses (`z_mu`, `z_log_var`) to sample `z`
        - `z` is the encoding vector
    """
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs) #necessary to save also custom layers

    def call(self, inputs):
        z_mu, z_log_var = inputs
        batch = tf.shape(z_mu)[0]
        dim = tf.shape(z_mu)[1]

        epsilon = K.random_normal(shape=(batch, dim))

        z = z_mu + K.exp(0.5 * z_log_var) * epsilon
        
        return z
  