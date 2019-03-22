import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
def swish(x):
    return (K.sigmoid(x) * x)
tf.keras.utils.get_custom_objects().update({'swish': swish})

class CNN:
    def __init__(self, input_dim=(28,28,1),
                 filter_size=[3],
                 output_dim=2,
                 activation=["relu"],
                 filters=[32],
                 embedding_dim = 32,
                 dropout = 0.2):
        if len(activation) == 1 and len(filter_size) > 1 :
            activation = activation * len(filter_size)
        if len(filters) == 1 and len(filter_size) > 1 :
            filters = filters * len(filter_size)
        assert len(activation) == len(filters) == len(filter_size) , "pass parameters properly"
        #build model here
        self.model = tf.keras.Sequential()
        # i = 1
        for j in range(len(filter_size)):
            self.model.add(tf.keras.layers.Conv2D(filters=filters[j],
                                             kernel_size=(filter_size[j],filter_size[j]),
                                             padding='same',
                                             activation=activation[j],
                                             input_shape=input_dim))
            # self.model.add(tf.keras.layers.Conv2D(filters=filters[j],
            #                                       kernel_size=(1, filter_size[j]),
            #                                       padding='same',
            #                                       activation=activation[j],
            #                                       input_shape=input_dim))
            if j%2 == 1:
                self.model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
                self.model.add(tf.keras.layers.Dropout(dropout))
                # self.model.add(tf.keras.layers.BatchNormalization(axis = 1))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(embedding_dim, activation='relu'))
        self.model.add(tf.keras.layers.Dense(output_dim,
                                        activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    # def __str__(self):
    #     for l in self.layers:
    #         print(l)
    #     return

    def fit(self,x_train,train_label, epoc = 20,batch_size=32,
            checkpoint_path="mnist.ckpt"):
        # x_train = tf.data.Dataset.from_tensor_slices(x_train)
        # train_label = tf.data.Dataset.from_tensor_slices(train_label)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_best_only=True,
                                                         monitor='val_loss',
                                                         mode='min',
                                                         save_weights_only=False,
                                                         verbose=1)
        self.model.fit(x_train,
                       train_label,
                       batch_size = batch_size,
                       epochs = epoc,
                       validation_split= 0.2,
                       callbacks= [cp_callback])


    def predict(self, input_x,model = "mnist.ckpt"):
        # m = tf.keras.models.load_model(model)
        return np.argmax(self.model.predict(input_x),axis=1)
    def evaluate(self,x_test,y_test):
        return self.model.evaluate(x_test,y_test)

    def get_embedding(self,x_test,model="mnist.h5"):
        m = tf.keras.models.load_model(model)
        emb_model = tf.keras.Model(inputs=m.input, outputs=m.layers[-2].output)
        return emb_model.predict(x_test)

if __name__ == "__main__":
    net = NN()
    input_x = np.asarray([[1, 0], [1, 1], [0, 1], [0, 0]])
    y_train = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]])

    # input_x = np.asarray([[1,0]])
    # y_train = np.asarray([[0,1]])
    l = net.fit_batch(input_x, y_train, epochs=200)
    # print(net.predict(input_x))
    plt.plot(l)
    plt.show()
