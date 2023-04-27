import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from deep.common import get_dataset_IJCE
import tensorflow


class LayerConfiguration:
    def __init__(self, size, activation_function="relu"):
        self.size = size
        self.activation_function = activation_function


train, validation = get_dataset_IJCE(True)
train: pd.DataFrame
validation: pd.DataFrame
print("Done loading data.")

#for i in range(1):
#    train = train.append(train)
print(len(train.index))
input_layer = Input(shape=8, name="input")

layers = [LayerConfiguration(32), LayerConfiguration(32)]
lr = input_layer
i = 0
for layer in layers:
    lr = Dense(layer.size, layer.activation_function, name=f"denselayer{i}")(lr)
    i += 1
output_layer = Dense(1, activation="sigmoid", name="Output")(lr)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

model.compile(optimizer='adam', loss=tensorflow.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(x=train.iloc[:, :-1], y=train.iloc[:, -1], epochs=100, batch_size=64)
