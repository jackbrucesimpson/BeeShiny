rm -rf train_lmdb val_lmdb
​
#Setup environment variables
TOOLS=~/caffe/build/tools
DATA_ROOT=~/Desktop/beeshiny/
​
#Create the training database
$TOOLS/convert_imageset \
--shuffle --gray \
$DATA_ROOT \
$DATA_ROOT/train.txt \
$DATA_ROOT/train_lmdb

#Create the validation database
$TOOLS/convert_imageset \
--shuffle --gray \
$DATA_ROOT \
$DATA_ROOT/val.txt \
val_lmdb
​
#Create the mean image database
$TOOLS/compute_image_mean train_lmdb $DATA_ROOT/mean.binaryproto
