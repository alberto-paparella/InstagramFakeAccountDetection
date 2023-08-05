import pandas as pd
import tensorflow.python.keras.losses
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from deep.common import get_dataset_instafake, LayerConfiguration
from tensorflow import convert_to_tensor
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.callbacks import ReduceLROnPlateau


def run_model(train):
    train: pd.DataFrame
    input_layer = Input(shape=len(train.columns) - 1, name="input")
    learning = {"rate": 0.001, "epochs": 100, 'batch_size': 16}
    layers = [LayerConfiguration(32), LayerConfiguration(32)]
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
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=4, min_lr=0, monitor="loss")
    data = model.fit(x=train.iloc[:, :-1], y=train.iloc[:, -1], epochs=learning["epochs"],
                     batch_size=learning["batch_size"], verbose=True, callbacks=[reduce_lr])
    return model, data.history, layers, learning
