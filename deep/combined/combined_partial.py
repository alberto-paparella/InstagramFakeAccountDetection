import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from deep.common import LayerConfiguration
import tensorflow
from tensorflow import convert_to_tensor

def run_model(train):

    train: pd.DataFrame
    x = convert_to_tensor(train.iloc[:, :-1])
    y = convert_to_tensor(train.iloc[:, -1])
    input_layer = Input(shape=len(train.columns)-1, name="input")

    layers = [LayerConfiguration(32), LayerConfiguration(32)]
    lr = input_layer
    i = 0
    for layer in layers:
        lr = Dense(layer.size, layer.activation_function, name=f"denselayer{i}")(lr)
        i += 1
    output_layer = Dense(1, activation="sigmoid", name="Output")(lr)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', loss=tensorflow.losses.BinaryCrossentropy(), metrics=["accuracy",
                                                                                          Precision(),
                                                                                          Recall()])
    data = model.fit(x=x, y=y, epochs=100, batch_size=64, verbose=True)
    return model, data.history, layers
