wget http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
wget http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz

mkdir -p ./data/deeplab_v3/mobilenet
mkdir -p ./data/deeplab_v3/xception
tar xvzf deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz -C ./data/deeplab_v3/mobilenet --strip=1
tar xvzf deeplabv3_pascal_train_aug_2018_01_04.tar.gz -C ./data/deeplab_v3/xception --strip=1

rm deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
rm deeplabv3_pascal_train_aug_2018_01_04.tar.gz