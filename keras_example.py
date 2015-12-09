from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(64, input_dim=20, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform', activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, init='uniform', activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
