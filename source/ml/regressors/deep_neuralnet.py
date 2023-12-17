#!usr/bin/env python

# Internal libraries
from source.ml.regressors import KerasRegressor


# External libraries
from tensorflow.keras.layers import (
    Layer, Dropout, SpatialDropout1D, Masking,
    Input, Dense, LSTM, Embedding, Bidirectional
)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import Model, Sequential, clone_model



tf_line_width = 120  # number of characters wide to print reports


# recurrent-neural network
class LSTMRegressor(KerasRegressor):
    def __init__(
        self,
        bidirectional: bool = False,        # learn both forward / backwards sequences
        mask_value: float = -1,             # padded feature values to be ignored
        units: int = 20,                    # hidden cells to remember state
        epochs: int = 10,
        patience: int = 1,
        batch_size: int = 16,
        dropout_rate: float = 0.2,          # % neurons to kill to help generalize
        regularizer_rate: float = 0.0,      # help generalize, none by default
        learning_rate: float = 0.001,
        learning_damper: float = 0.0001,
        validation_split: float = 0.1,
        random_state: int = None,           # determinism
        verbose: int = 1,                   # monitoring & troubleshooting
        **kwargs
    ):
        super().__init__(
            random_state=random_state,
            bidirectional=bidirectional,
            mask_value=mask_value,
            n_timesteps=kwargs.get('n_timesteps'),    # deferred to fit()
            n_features=kwargs.get('n_features'),      # deferred to fit()
            units=units,
            epochs=epochs,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            regularizer_rate=regularizer_rate,
            learning_rate=learning_rate,
            callbacks=kwargs.get('callbacks', [
                LearningRateScheduler(
                    schedule=lambda epoch, learning_rate: learning_rate / (1 + learning_damper * epoch),
                ),
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.0001,
                    patience=patience,
                    restore_best_weights=True,
                    verbose=verbose,
                )
            ]),
            validation_split=validation_split,
            loss='mean_squared_error',                     # regression
            optimizer=kwargs.get('optimizer', 'rmsprop'),  # needed cloning in parallel
        )
        return

    # Accepts input X of shape: (samples, n_timesteps, n_features)
    def fit(self, X, y, **kwargs):
        super().set_params(
            n_timesteps=X.shape[1],
            n_features=X.shape[2],
        )
        return super().fit(X, y, **kwargs)

    # Neural network architecture
    def _keras_build_fn(
        self,
        n_timesteps: int,
        n_features: int,
        bidirectional: bool,
        mask_value: float,
        learning_rate: float,
        dropout_rate: float,
        regularizer_rate: float,
        units: int,
    ) -> Model:
        model = Sequential(name='rnn-sequential')

        # prevent padded time steps to be interpreted as noise
        model.add(Masking(
            mask_value=mask_value,
            input_shape=(n_timesteps, n_features))
        )
        lstm = LSTM(
            units=units,
            recurrent_dropout=dropout_rate,
            kernel_regularizer=l1_l2(l1=0, l2=regularizer_rate),
        )
        model.add(
            Bidirectional(lstm) if bidirectional else lstm,
            name='lstm',
        )
        model.add(Dense(
            name='regressor',
            units=1,
            activation='sigmoid',
        ))
        model.compile(
            loss=self.loss,
            optimizer=RMSprop(learning_rate=learning_rate),
        )
        if self.verbose > 0:
            print(model.summary(line_length=tf_line_width))
        return model
