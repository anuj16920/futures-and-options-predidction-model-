import tensorflow as tf
print('TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPU devices:', gpus)
print('Built with CUDA:', tf.test.is_built_with_cuda())
if gpus:
    print('GPU Name:', gpus[0].name)
    print('GPU detected successfully!')
else:
    print('No GPU detected')
