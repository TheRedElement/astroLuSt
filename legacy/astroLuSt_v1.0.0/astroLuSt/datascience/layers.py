
#%%imports
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from typing import Tuple


#%%definitions
class Sampling(Layer):
    """
        - sampling layer as found in Variational Autoencoders
        - uses (`z_mu`, `z_log_var`) to sample `z`
        - `z` is the encoding vector

        Attributes
        ----------

        Methods
        -------
            - `call()`

        Dependencies
        ------------
            - `keras`
            - `tensorflow`

        Comments
        --------

    """
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs) #necessary to save also custom layers

    def call(self,
        inputs:Tuple[tf.Tensor]
        ) -> tf.Tensor:
        """
            - call method of the layer

            Parameters
            ----------
                - `inputs`
                    - `Tuple[tf.Tensor]`
                    - contains two entries
                        - `z_mu`
                            - means of the latent variables
                        - `z_log_var`
                            - log of the variances of the latent variables

            Raises
            ------

            Returns
            -------
                - `z`
                    - `tf.Tensor`
                    - random sample from the latent space
                    - created by means of reparametrization trick

            Comments
            --------
        """

        z_mu, z_log_var = inputs
        batch = tf.shape(z_mu)[0]
        dim = tf.shape(z_mu)[1]

        epsilon = K.random_normal(shape=(batch, dim))

        z = z_mu + K.exp(0.5 * z_log_var) * epsilon
        
        return z
