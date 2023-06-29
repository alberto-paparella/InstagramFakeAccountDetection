import pandas as pd
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from deep.common import get_dataset_spz, LayerConfiguration
import tensorflow
from tensorflow.keras.metrics import Accuracy, Precision, Recall


def run_model(train):
    train: pd.DataFrame
    input_layer = Input(shape=len(train.columns) - 1, name="input")

    layers = [LayerConfiguration(32), LayerConfiguration(32)]
    lr = input_layer
    i = 0
    for layer in layers:
        lr = Dense(layer.size, layer.activation_function, name=f"denselayer{i}")(lr)
        i += 1
    output_layer = Dense(1, activation="sigmoid", name="Output")(lr)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=tensorflow.losses.BinaryCrossentropy(), metrics=["accuracy",
                                                                                                    Precision(),
                                                                                                    Recall()])
    model.fit(x=train.iloc[:, :-1], y=train.iloc[:, -1], epochs=100, batch_size=8, verbose=True)
    return model
