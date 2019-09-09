import tensorflow as tf


class DataLoader:

    def __init__(self, config, image_preprocessing_fn, image_size):
        # make interface from the file names to the input tensor
        # contain the loaded file list and the GT

        self.image_path = tf.placeholder(tf.string, [None])
        img_list = []
        for i in range(config['batch_size']):
            file_data = tf.read_file(self.image_path[i])
        # Decode the image data
            img = tf.image.decode_jpeg(file_data, channels=3)
            img = tf.to_float(img)
            # normalize the image
            if config['normalize']:
                img = img/255.0
            image = image_preprocessing_fn(img, image_size, image_size)
            img_list.append(image)
        self.input_images = tf.stack(img_list, 0)
