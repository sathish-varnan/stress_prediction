import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential


stress_data = pd.read_csv("./data/stress.csv")

stress_data.drop(columns=["Participant", "Time(sec)"], inplace=True)

features, labels = stress_data[["HR", "respr"]].values, stress_data["Label"].values

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

model = Sequential()

model.add(Dense(units=32, activation="relu", input_dim=2))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

prediction = model.evaluate(x_test, y_test)

print(f"Accuracy : {prediction[1]}")

model.save("stress_predictor.h5")