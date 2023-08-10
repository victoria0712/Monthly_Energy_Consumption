import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input
from ispots.forecaster.utils.feature_builder import feature_builder, generate_sequence, convert_to_tf_dataset

class LSTMModel():
    """ Deep learning model which uses LSTM to generate forecast prediction

    Parameters
    ----------
    freq: str, default='30min'
        Frequency of the time series
    
    horizon: str, default='1d'
        Forecast horizon

    epochs: int, default=1000
        Number of epochs to train the model

    batch_size: int, default=64
        Number of samples per gradient update

    lr: float, default=0.001
        Initial learning rate of the optimizer

    type: {'vanilla', 'encoder-decoder'}, default='vanilla'
        Type of LSTM architecture

    stateful: bool, default=True
        Determines if the model is stateless or stateful. Only applicable when type='encoder-decoder'.
    
    lstm_units: int, default=50
        Number of neurons in the LSTM layer

    lstm_layers: int, default=1
        Number of LSTM layers. Only applicable when type='vanilla'.

    dense_units: int, default=50
        Number of neurons in the dense layer. Only applicable when type='vanilla'.

    lstm_layers: int, default=1
        Number of dense layers. Only applicable when type='vanilla'.
    
    cyclic_feature_encoding: {'sincos', 'onehot'}, default='sincos'
        Cyclic feature encoding method

    val_size: float, default=0.05
        Proportion of dataset to be used as validation set

    early_stopping_patience: int, default=30
        Number of epochs with no improvement after which training will be stopped
    
    lr_scheduler_patience: int, default=20
        Number of epochs with no improvement after which learning rate will be reduced

    factor: float, default=0.5
        Factor by which the learning rate will be reduced. new_lr = lr * factor.
    """
    def __init__(
        self, 
        freq='30min', 
        horizon='1d',
        epochs=1000,
        batch_size=64,
        lr = 0.001,
        type = 'vanilla',
        stateful = True,
        lstm_units = 50,
        dense_units = 50,
        lstm_layers = 1,
        dense_layers = 1,
        cyclic_feature_encoding='sincos',
        val_size = 0.05,
        early_stopping_patience = 30,
        lr_scheduler_patience = 20,
        factor = 0.5,
    ):
        if pd.Timedelta(freq) < pd.Timedelta('1d') and pd.Timedelta('1day') % pd.Timedelta(freq) != pd.Timedelta(0):
            raise ValueError(f'{freq} is not daily divisable')
        elif pd.Timedelta(freq) > pd.Timedelta('1d'):
            raise ValueError(f'{freq} frequency not supported. Only support daily or daily divisable frequency')

        if cyclic_feature_encoding not in ['sincos', 'onehot']:
            raise ValueError("Supported cyclic_feature_encoding methods are: ['sincos', 'onehot']")
        if type not in ['vanilla', 'encoder-decoder']:
            raise ValueError("Supported LSTM types are: ['vanilla', 'encoder-decoder']")

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.lstm_layers = lstm_layers
        self.dense_layers = dense_layers
        self.type = type
        self.stateful = stateful
        self.val_size = val_size
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.factor = factor

        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()

        self.freq = freq
        self.horizon = horizon
        self.cyclic_feature_encoding = cyclic_feature_encoding
        self.look_back_cycle = 4 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else 14
        self.cycle_periods = 7 if pd.Timedelta(self.freq) == pd.Timedelta('1d') else int(pd.Timedelta('1d') / pd.Timedelta(self.freq))
        self.look_back_periods = self.look_back_cycle * self.cycle_periods
        self.fcst_periods = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.freq))
        self.name = 'LSTM'

    def fit(self, X):
        """ Generate features for data and fit model with input data and features

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, 1)
            Univariate time series dataframe from preprocessor

        Returns
        -------
        self: object
            Fitted model
        """
        X = X.copy()
        X_train = X.iloc[:-int(self.val_size*len(X))]
        X_val = X.iloc[-int(self.val_size*len(X))-self.look_back_periods:]
        X_train_time = feature_builder(X_train, self.freq, self.name, self.cyclic_feature_encoding).values
        X_val_time = feature_builder(X_val, self.freq, self.name, self.cyclic_feature_encoding).values

        self.scaler_input.fit(X_train.iloc[:-self.fcst_periods])
        X_train_scaled = self.scaler_input.transform(X_train)
        X_train_total = np.concatenate([X_train_scaled, X_train_time], axis=1)
        X_val_scaled = self.scaler_input.transform(X_val)
        X_val_total = np.concatenate([X_val_scaled, X_val_time], axis=1)

        self.scaler_output.fit(X_train.iloc[self.look_back_periods:])
        y_train_scaled = self.scaler_output.transform(X_train)
        y_val_scaled = self.scaler_output.transform(X_val)

        X_train, _ = generate_sequence(X_train_total, self.look_back_periods, self.fcst_periods)
        X_val, _ = generate_sequence(X_val_total, self.look_back_periods, self.fcst_periods)
        _, y_train = generate_sequence(y_train_scaled, self.look_back_periods, self.fcst_periods)
        _, y_val = generate_sequence(y_val_scaled, self.look_back_periods, self.fcst_periods)

        train_dataset = convert_to_tf_dataset(X_train, y_train, self.batch_size)
        val_dataset = convert_to_tf_dataset(X_val, y_val, self.batch_size)

        tf.random.set_seed(2022)

        if self.type == 'vanilla':
            # Model 1: Vanilla LSTM
            model = Sequential()
            for _ in range(self.lstm_layers-1):
                model.add(LSTM(self.lstm_units, return_sequences=True,))
            model.add(LSTM(self.lstm_units, return_sequences=False,))
            
            for _ in range(self.dense_layers-1):
                model.add(Dense(self.dense_units, activation='relu'))
            model.add(Dense(self.fcst_periods))

        elif self.type == 'encoder-decoder' and self.stateful == False:
            ## Model 2: Encoder-Decoder LSTM
            model = Sequential()
            # encoder
            model.add(LSTM(self.lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False,))
            model.add(RepeatVector(self.fcst_periods))

            # decoder
            model.add(LSTM(self.lstm_units, return_sequences=True))

            model.add(TimeDistributed(Dense(1)))

        elif self.type == 'encoder-decoder' and self.stateful == True:
            # Model 3: Stateful Encoder-Decoder LSTM
            encoder_inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
            encoder_l1 = LSTM(self.lstm_units, return_state=True)
            encoder_outputs1 = encoder_l1(encoder_inputs) # [hidden state output fori last timestep, hidden state output for last timestep, internal cell state for last time step]
            encoder_states1 = encoder_outputs1[1:]

            decoder_inputs = RepeatVector(self.fcst_periods)(encoder_outputs1[0])
            decoder_l1 = LSTM(self.lstm_units, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
            decoder_outputs1 = TimeDistributed(tf.keras.layers.Dense(1))(decoder_l1)

            model = Model(encoder_inputs,decoder_outputs1)


        early_stopping = tf.keras.callbacks.EarlyStopping(mode='min', 
                                                        patience=self.early_stopping_patience, 
                                                        restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=self.factor, 
                                                            patience=self.lr_scheduler_patience)

        model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(lr=self.lr),
                    metrics=[tf.metrics.RootMeanSquaredError()])

        history = model.fit(train_dataset, epochs=self.epochs, validation_data=val_dataset, callbacks=[early_stopping, lr_scheduler])

        self.history = history
        self.model = model

        return self


    def predict(self, X):
        """ Generate forecast predictions using fitted model

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, 1)
            The data used to generate forecast predictions.

        Returns
        -------
        pandas.DataFrame
            Time series dataframe containing predictions for the forecast horizon
        """
        X = X.copy()
        features = feature_builder(X, self.freq, self.name, self.cyclic_feature_encoding).values
        X_scaled = self.scaler_input.transform(X)
        X_total = np.concatenate([X_scaled, features], axis=1)
        
        test = X_total[-self.cycle_periods * self.look_back_cycle:]
        test = test.reshape((1, test.shape[0], test.shape[1]))
        pred = self.model.predict(test)
        if self.type == 'vanilla':
            fcst = self.scaler_output.inverse_transform(pred)[0]
        else:
            fcst = self.scaler_output.inverse_transform(pred[0]).flatten()

        start_time = X.index[-1] + pd.Timedelta(self.freq)
        df_fcst = pd.DataFrame({'ds': pd.date_range(start=start_time, periods=self.fcst_periods, freq=self.freq), 'y': fcst})
        df_fcst.set_index('ds', inplace=True)

        return df_fcst
