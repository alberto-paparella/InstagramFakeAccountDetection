import pandas as pd
from tensorflow.keras.layers import Input, Dense, Normalization
from tensorflow.keras.models import Model
from deep.common import LayerConfiguration
from tensorflow import convert_to_tensor
import tensorflow as tf

def run_model(train):
    train: pd.DataFrame
    x = convert_to_tensor(train.iloc[:, :-1])
    y = convert_to_tensor(train.iloc[:, -1])

    #for i in range(1):
    #    train = train.append(train)
    input_layer = Input(shape=len(train.columns)-1, name="input")

    layers = [LayerConfiguration(32),LayerConfiguration(32)]
    lr = input_layer
    i = 0
    for layer in layers:
        lr = Dense(layer.size, layer.activation_function, name=f"denselayer{i}")(lr)
        i += 1
    output_layer = Dense(1, activation="sigmoid", name="Output")(lr)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    model.fit(x=x, y=y, epochs=100, batch_size=64, validation_split=0.2, verbose=False)
    return model