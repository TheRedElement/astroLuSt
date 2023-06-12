
#%%imports
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay, ExponentialDecay
from tqdm import tqdm

from sklearn.base import BaseEstimator

#%%definitions
class PeriodLSTM(Model, BaseEstimator):
    def __init__(self,
        input_shape, lstm_units, hidden_units, out_shape,
        #loss function
        loss:str='mse',
        #optmizer
        optimizer:str='adam', learning_rate:float=1E-3, learning_rate_decay:float=0.99,
        #training
        epochs:int=10,        
        #misc
        model_name:str='period_lstm',
        verbose:int=0,
        **kwargs       
    ):
        super(PeriodLSTM, self).__init__()

        #architecture
        self.in_shape       = input_shape
        self.lstm_units     = lstm_units
        self.hidden_units   = hidden_units
        self.out_shape      = out_shape

        #loss function
        self.loss = loss
        
        #optimizer
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        #training
        self.epochs = epochs

        #misc
        self.model_name = model_name
        self.verbose = verbose

        return  
        
    def build_model(self):
        self.lstm_layers = []
        self.hidden_layers = []
        
        self.input_layer = Input(self.in_shape)

        x = self.input_layer

        for idx, lstm_units in enumerate(self.lstm_units):
            if idx < (len(self.lstm_units)-1):
                x = LSTM(units=lstm_units, return_sequences=True, activity_regularizer='l2')(x)
            else:
                x = LSTM(units=lstm_units, return_sequences=False, activity_regularizer='l2')(x)
        for hunits in self.hidden_units:
            x = Dense(units=hunits, activation='relu')(x)
        
        #output layer
        self.output_layer = Dense(units=self.out_shape, activation='linear')(x)

        self.model = Model(inputs=self.input_layer, outputs=self.output_layer, name=self.model_name)
        
        return

    def compile_model(self,
        compile_kwargs:dict=None
        ):
       
        if compile_kwargs is None:
            compile_kwargs={}

        #setup optimizer
        lr = ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.epochs,
            decay_rate=self.learning_rate_decay,
        )
        if self.optimizer == 'adam': opt = Adam(learning_rate=lr)
        elif self.optimizer == 'sgd': opt = SGD(learning_rate=lr)


        #compile
        self.model.compile(
            optimizer=opt, loss=self.loss,
            metrics=[
                'accuracy'
            ],
            **compile_kwargs
        )        

        return
    
    """FOR TIMESERIES OF VARYING LENGTHS
    @tf.function
    def train_step(self,
        X:np.ndarray, y:np.ndarray
        ):
        
        with tf.GradientTape() as tape:
            # y_pred = self.model.predict(X)
            y_pred = self.model(X, training=True)
            loss = self.model.compiled_loss(y, y_pred)
        
        #compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        #update weights
        self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))

        #update metrics
        self.model.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.model.metrics}

    def fit_model(self,
        X_batches:np.ndarray, y_batches:np.ndarray,
        epochs:int,
        ):

        # self.model.fit(X, y)
        for epoch in range(epochs):

            for bidx, (X_batch, y_batch) in enumerate(tqdm(zip(X_batches, y_batches))):
                X_batch = np.array([*X_batch]).astype(np.float32)
                y_batch = np.array([*y_batch]).astype(np.float32)

                metrics = self.train_step(X_batch, y_batch)
                # self.model.fit(X_batch, y_batch)
                print(X_batch.shape)

            #epoch summary
            printmetrics = '    '
            for k, v in metrics.items(): printmetrics += f'{k}: {v}, '
            print(f'Epoch {epoch+1}/{epochs}, Batch {bidx+1}/{len(X_batches)}')
            print(printmetrics)

        return
    """
    
    def score(self,
        X:np.ndarray, y:np.ndarray=None
        ):
        
        return 4
# %%
