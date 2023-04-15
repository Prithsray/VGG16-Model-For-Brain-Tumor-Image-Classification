# -*- coding: utf-8 -*-


# vgg16 model used for transfer learning on the dogs and cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten,MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
import keras
from sklearn.metrics import auc

# define cnn model
def define_model():
	# load model
	model = VGG16(include_top=False, input_shape=(64, 64, 3),weights='imagenet')
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(256, activation='relu', kernel_initializer='he_uniform')(flat1)
	class2 = Dense(512, activation='sigmoid', kernel_initializer='he_uniform')(class1)
	class3 = Dense(256, activation='sigmoid', kernel_initializer='he_uniform')(class2)
	#class4 = Dense(512, activation='sigmoid', kernel_initializer='he_uniform')(class3)

	#class2 = MaxPooling2D(31,31,32)(class1)
	output = Dense(1, activation='sigmoid')(class3)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = Adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Precision(),keras.metrics.Recall(),keras.metrics.AUC(),keras.metrics.TruePositives(),keras.metrics.TrueNegatives(),keras.metrics.FalseNegatives(),keras.metrics.FalsePositives()])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

# run the test harness for evaluating a model
	# define model
model = define_model()
	# create data generator
datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterator
train_it = datagen.flow_from_directory('D:/COLLEGEMATERIALS/Research Paper Work/archive (6)/Training',
		class_mode='binary', batch_size=64, target_size=(64, 64))
test_it = datagen.flow_from_directory('D:/COLLEGEMATERIALS/Research Paper Work/archive (6)/Testing',
		class_mode='binary', batch_size=64, target_size=(64, 64))
# fit model
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)
	# evaluate model
_,acc,precison,rcall,aucc,truepo,truene,falsepo,falsene = model.evaluate(test_it, steps=len(test_it), verbose=0)
	#print('> %.3f' % (acc * 100.0))
	# learning curves
ac= (truepo+truene)/(truene+truepo+falsene+falsepo)

print("Accuracy ->",ac)
pre=truepo/(truepo+falsepo)
print("Precision-> ",pre)

recall=truepo/(truepo + falsene)
print("Recall-> ",recall)
specificity=truene/(truene+falsepo)
print("Specificity -> ",specificity)
f1= 2*pre*recall/(pre+recall)
print("F1-Score -> ",f1)


fpr= falsepo/(falsepo+truene)


a=1-specificity
#pyplot.plot(history[keras.metrics.AUC()], color='blue')
#pyplot.show()
#auc =recall+se
#auc_score = auc(recall, pre)
auc = (1/2)- (fpr/2) + (recall/2)
print(auc)
summarize_diagnostics(history)

# entry point, run the test harness
