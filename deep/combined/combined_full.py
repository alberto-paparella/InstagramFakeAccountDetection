import pandas as pd
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Normalization
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from deep.common import LayerConfiguration
from tensorflow import convert_to_tensor
import tensorflow


def run_model(train):
    train: pd.DataFrame
    x = convert_to_tensor(train.iloc[:, :-1])
    y = convert_to_tensor(train.iloc[:, -1])
    input_layer = Input(shape=len(train.columns) - 1, name="input")
    learning = {"rate": 0.001, "epochs": 100, 'batch_size': 128}
    layers = [LayerConfiguration(64), LayerConfiguration(64)]
    lr = input_layer
    i = 0
    for layer in layers:
        lr = Dense(layer.size, layer.activation_function, name=f"denselayer{i}")(lr)
        i += 1
    output_layer = Dense(1, activation="sigmoid", name="Output")(lr)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=learning["rate"]), loss=tensorflow.losses.BinaryCrossentropy(),
                  metrics=["accuracy",
                           Precision(),
                           Recall()])
    data = model.fit(x=train.iloc[:, :-1], y=train.iloc[:, -1], epochs=learning["epochs"],
                     batch_size=learning["batch_size"], verbose=True)
    return model, data.history, layers, learning
