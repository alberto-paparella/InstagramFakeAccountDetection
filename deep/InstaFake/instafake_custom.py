import pandas as pd
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from deep.common import LayerConfiguration
import tensorflow
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ReduceLROnPlateau


def run_model(train):
    train: pd.DataFrame
    input_layer = Input(shape=len(train.columns) - 1, name="input")
    learning = {"rate": 0.001, "epochs": 100, 'batch_size': 16}
    layers = [LayerConfiguration(8), LayerConfiguration(8)]
    lr = input_layer
    i = 0
    for layer in layers:
        lr = Dense(layer.size, layer.activation_function, name=f"denselayer{i}")(lr)
        i += 1
    output_layer = Dense(1, activation="sigmoid", name="Output")(lr)
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=4, min_lr=0, monitor="loss")
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=learning["rate"]), loss=tensorflow.losses.BinaryCrossentropy(),
                  metrics=["accuracy",
                           Precision(),
                           Recall()])
    data = model.fit(x=train.iloc[:, :-1], y=train.iloc[:, -1], epochs=learning["epochs"],
                     batch_size=learning["batch_size"], verbose=True, callbacks=[reduce_lr], validation_split=0.1)
    return model, data.history, layers, learning
