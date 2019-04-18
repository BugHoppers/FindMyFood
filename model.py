from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.applications import InceptionV3


def buildmodel(N):

    base_model = InceptionV3(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    classifier = base_model.output
    classifier = GlobalAveragePooling2D()(classifier)
    classifier = Dense(2048, activation='relu')(classifier)
    classifier = Dense(1024, activation='relu')(classifier)
    classifier = Dense(512, activation='relu')(classifier)

    pred = Dense(N, activation='softmax')(classifier)

    model = Model(input=base_model.input, output=pred)

    for layer in model.layers[:229]:
        layer.trainable = False
    for layer in model.layers[229:]:
        layer.trainable = True

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model
