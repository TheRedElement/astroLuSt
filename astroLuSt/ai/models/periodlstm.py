
#%%imports
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from sklearn.base import BaseEstimator

#%%definitions
class PeriodLSTM(BaseEstimator):
    def __init__(self, input_shape, hidden_units, output_units):
        
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.output_units = output_units
        
        self.model = self.build_model()

    def build_model(self):
        # Define the input layer
        inputs = Input(shape=self.input_shape)

        # Add the LSTM layer
        x = LSTM(units=self.hidden_units)(inputs)

        # Add the output layer
        outputs = Dense(units=self.output_units, activation='relu')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def score(self,
        X:np.ndarray, y:np.ndarray=None
        ):
        
        return