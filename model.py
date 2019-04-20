from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.applications import MobileNet

from preprocess_data import generator


def buildmodel(N):

    base_model = MobileNet(weights='imagenet',include_top=False, input_shape=(224, 224, 3))

    classifier = base_model.output
    classifier = GlobalAveragePooling2D()(classifier)
    classifier = Dense(2048, activation='relu')(classifier)
    classifier = Dense(1024, activation='relu')(classifier)
    classifier = Dense(512, activation='relu')(classifier)

    pred = Dense(N, activation='softmax')(classifier)

    model = Model(input=base_model.input, output=pred)

    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train(dir):
    train_generator = generator(dir)
    no_of_classes = len(train_generator.class_indices)

    model = buildmodel(no_of_classes)

    model.fit_generator(train_generator, steps_per_epoch=len(
        train_generator)/64, epochs=10)
    
    model.save('model.h5')

    return model