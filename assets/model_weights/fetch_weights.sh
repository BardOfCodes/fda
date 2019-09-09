# Mini script to fetch all the weights for different networks used in our work,
# unzip and rename them for consistency.

# VGG 16
mkdir vgg_16
cd vgg_16
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt model.ckpt
cd ..

# Inception v3
mkdir inception_v3
cd inception_v3
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvf inception_v3_2016_08_28.tar.gz
mv inception_v3.ckpt model.ckpt
cd ..

# Resnet V1 152
mkdir resnet_v1_152
cd resnet_v1_152
wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
tar -xvf resnet_v1_152_2016_08_28.tar.gz
mv resnet_v1_152_2016_08_28.ckpt model.ckpt
cd ..

# Inception Resnet V2
mkdir inception_resnet_v2
cd inception_resnet_v2
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
tar -xvf inception_resnet_v2_2016_08_30.tar.gz
mv inception_resnet_v2_2016_08_30.ckpt model.ckpt
cd ..


# PNASNet Large
mkdir pnasnet_5_large_331
cd pnasnet_5_large_331
wget https://storage.googleapis.com/download.tensorflow.org/models/pnasnet-5_large_2017_12_13.tar.gz
tar -xvf pnasnet-5_large_2017_12_13.tar.gz
mv pnasnet-5_large_2017_12_13.ckpt model.ckpt
cd ..