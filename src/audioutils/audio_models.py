import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout

from sklearn.metrics.pairwise import cosine_similarity


# Implementación de la métrica de Franco
class LuqueMetric():
    def metric_np(self, y_true, y_pred=[]):
        S = cosine_similarity(y_true, y_pred)
        S2 = S + 1
        M = S2.diagonal() / S2.sum(axis=0)
        return -M.mean()

    def metric_tf(self, y_true, y_pred=[]):
        # Calcular cosine similarity matrix S2
        y_true = K.l2_normalize(y_true, axis=-1)
        y_pred = K.l2_normalize(y_pred, axis=-1)
        S = K.dot(y_true, tf.matrix_transpose(y_pred))

        # Llevar valores a [0, 2]
        S2 = S + 1
        M = tf.diag_part(S2) / K.sum(S2, axis=0)
        return -K.mean(M)

    # Quiero que 1 sea el valor máximo de la función.
    # La función da 0 cuando los vectores son ortogonales
    def vector(self, y_true, y_pred=[]):
        metric = self.metric_np(y_true, y_pred)
        maximo = self.metric_np(y_true, y_true)
        minimo = self.metric_np(y_pred, -y_pred)
        return (metric - minimo) / (maximo - minimo)

    def tensor(self, y_true, y_pred=[]):
        metric = self.metric_tf(y_true, y_pred)
        maximo = self.metric_tf(y_true, y_true)
        minimo = self.metric_tf(y_pred, -y_pred)
        return -tf.truediv(metric, maximo)


class ModelCNN(object):
    def cnn_melspect_1D(input_shape, dimensions):
        model = Sequential()

        # 1
        model.add(
            Conv1D(
                filters=128,
                kernel_size=3,
                input_shape=input_shape,
                activation='relu',
                kernel_initializer='normal',
                padding='valid'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2, strides=2))
        # model.add(Dropout(0.5))

        model.add(Flatten())

        # Regular MLP
        model.add(
            Dense(
                128,
                kernel_initializer='glorot_normal',
                bias_initializer='glorot_normal',
                activation='relu'))
        # model.add(Dropout(0.5))

        model.add(Dense(dimensions, kernel_initializer='normal'))
        return model
