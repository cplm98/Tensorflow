# from https://www.tensorflow.org/tutorials/keras/classification
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)

print(len(train_labels))

# show original image
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize data by scaling them to between 0 and 1
# Make sure to normalize test and training data in the same way

train_images = train_images / 255.0
test_images = test_images / 255.0

# Sample normalized images
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# Build the model
# layer is the basic building block of the network
# most deep learning comes from chaining simple layers togethers
model = tf.keras.Sequential([
    # flatten turns the 2D array of the image into a 1D array of length=28*28=784
    # no learning, just reformatting
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # 2 Dense layers which are fully connected. first with 128 nodes
    tf.keras.layers.Dense(128, activation='relu'),
    # output layer of 10 nodes outputs probability of each class
    tf.keras.layers.Dense(10)
])

# Compile Model
model.compile(
    optimizer = 'adam', # how model is updated based on loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # measures accuracy of the model, and steering during training
    metrics = ['accuracy'] # what it's using to measure success
)

# Train Model
model.fit(train_images, train_labels, epochs=10)

# Test Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# higher training set accuracy demonstrates overfitting
print('\nTest accuracy:', test_acc)

# trained model can now make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) # softmax layers converts logits to probabilities

predictions = probability_model.predict(test_images)

# use the highest probability class as prediction

#plot results
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
