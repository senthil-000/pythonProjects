import tensorflow as tf

class mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0/95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks=mycallbacks()
# get data online
data = tf.keras.datasets.fashion_mnist

# training data
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# Normalization
training_images = training_images / 255.0
test_images = test_images / 255.0

# model
model = tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28)),
         tf.keras.layers.Dense(128, activation=tf.nn.relu),
         tf.keras.layers.Dense(10, activation=tf.nn.softmax)]
)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])


classification = model.predict(test_images)
print(classification[0])
print(test_labels[0])
