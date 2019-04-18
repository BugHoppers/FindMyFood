from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=15,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.01,
                             zoom_range=[0.9, 1.25],
                             horizontal_flip=True,
                             vertical_flip=False,
                             fill_mode='reflect',
                             data_format='channels_last',
                             brightness_range=[0.5, 1.5],
                             rescale=1./255)


def generator(dir):
    generator = datagen.flow_from_directory(
        dir,
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')
    return generator
