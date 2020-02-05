import hashlib
from functools import partial

import matplotlib.pyplot as plt
import neptune
import numpy as np
from tensorflow import keras

from sources.model import get_model
from sources.parameters import PARAMS
from sources.utils import lr_scheduler, log_data, log_model_weights

# Select project
neptune.init('neptune-workshops/AII-Optimali')

# Create experiment
exp = neptune.create_experiment(name='classification_example',
                                tags=['classification', 'tf_2'],
                                upload_source_files=['4-train-classification.py', 'sources/*.py'],
                                params=PARAMS)
# Dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

exp.set_property('train_images_version', hashlib.md5(train_images).hexdigest())
exp.set_property('train_labels_version', hashlib.md5(train_labels).hexdigest())
exp.set_property('test_images_version', hashlib.md5(test_images).hexdigest())
exp.set_property('test_labels_version', hashlib.md5(test_labels).hexdigest())

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

exp.set_property('class_names', class_names)

for j, class_name in enumerate(class_names):
    plt.figure(figsize=(10, 10))
    label_ = np.where(train_labels == j)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[label_[0][i]], cmap=plt.cm.binary)
        plt.xlabel(class_names[j])
    exp.log_image('example_images', plt.gcf())

# Prepare model
model = get_model(PARAMS)

# Log model summary
model.summary(print_fn=lambda x: exp.log_text('model_summary', x))

# Train model
model.fit(train_images, train_labels,
          batch_size=PARAMS['batch_size'],
          epochs=PARAMS['n_epochs'],
          shuffle=PARAMS['shuffle'],
          callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_data(logs, exp)),
                     keras.callbacks.EarlyStopping(patience=PARAMS['early_stopping'],
                                                   monitor='accuracy',
                                                   restore_best_weights=True),
                     keras.callbacks.LearningRateScheduler(partial(lr_scheduler,
                                                                   neptune=exp,
                                                                   learning_rate=PARAMS['learning_rate']))]
          )

# Log model weights
log_model_weights(model, exp)

# Evaluate model
eval_metrics = model.evaluate(test_images, test_labels, verbose=0)
for j, metric in enumerate(eval_metrics):
    exp.log_metric('eval_' + model.metrics_names[j], metric)
