import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from PIL import Image
import json
np.set_printoptions(precision=3,suppress=True)
from utils import normalize, plot_result, optimize
from utils import optimize_resnet, optimize_SGD
model = keras.applications.VGG16(weights='imagenet',include_top=True)
classes = json.loads(open('imagenet_classes.json','r').read())

x = tf.Variable(tf.random.normal((1,224,224,3)))

plt.imshow(normalize(x[0]))


target = [49] # 'African crocodile, Nile crocodile, Crocodylus niloticus',

def total_loss(target,res):
    return 10*tf.reduce_mean(keras.metrics.sparse_categorical_crossentropy(target,res)) + \
           0.005*tf.image.total_variation(x,res)

optimize(x,target,loss_fn=total_loss)


x = tf.Variable(tf.random.normal((1,224,224,3)))
optimize(x,[340],loss_fn=total_loss) # zebra

x = tf.Variable(tf.random.normal((1,224,224,3)))
optimize(x,[63],loss_fn=total_loss) # Indian Cobra


#using resnet
x = tf.Variable(tf.random.normal((1,224,224,3)))
optimize_resnet(x,target=[49],loss_fn=total_loss)

#using different optimizer
x = tf.Variable(tf.random.normal((1,224,224,3)))
optimize_SGD(x,target=[49],loss_fn=total_loss)