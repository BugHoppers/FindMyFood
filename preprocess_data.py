from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input)


def generator(dir):
    generator = datagen.flow_from_directory(
        dir,
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')
    return generator
