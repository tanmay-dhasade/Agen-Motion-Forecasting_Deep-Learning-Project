#!/bin/bash

# Make a temporary download folder
TEMP_DOWNLOAD_DIR='/home/trdhasade/Dataset'
TEMP_DATASET_DIR='/home/trdhasade/ExtractedDataset'

# # Download sample zarr
# # echo "Downloading sample zarr dataset..."
# # wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/sample.tar \
# #     -q --show-progress -P $TEMP_DOWNLOAD_DIR
# echo "Start Extracting ..........."
# mkdir -p $TEMP_DATASET_DIR/scenes
# tar xf $TEMP_DOWNLOAD_DIR/train_2.tar -C $TEMP_DATASET_DIR/scenes
# echo "Train.tar done ................"

# # Download semantic map
# # echo "Downloading semantic map..."
# # wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/semantic_map.tar \
#     # -q --show-progress -P $TEMP_DOWNLOAD_DIR
# mkdir -p $TEMP_DATASET_DIR/semantic_map
# tar xf $TEMP_DOWNLOAD_DIR/semantic_map.tar -C $TEMP_DATASET_DIR/semantic_map
# cp $TEMP_DATASET_DIR/semantic_map/meta.json $TEMP_DATASET_DIR/meta.json
# echo "Semantic map done ........."

# # Download aerial maps
# # echo "Downloading aerial maps (this can take a while)..."
# # wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/aerial_map.tar \
#     # -q --show-progress -P $TEMP_DOWNLOAD_DIR
# tar xf $TEMP_DOWNLOAD_DIR/aerial_map.tar -C $TEMP_DATASET_DIR
# echo "aerial_map done .........."
# # Dowload sample configuration
# # wget https://raw.githubusercontent.com/lyft/l5kit/master/examples/visualisation/visualisation_config.yaml -q
# # wget https://raw.githubusercontent.com/lyft/l5kit/master/examples/RL/gym_config.yaml -q

# # Install L5Kit
# echo "Installing L5kit..."
# pip install --progress-bar off --quiet -U l5kit pyyaml

echo "Dataset and L5kit are ready !"
echo $TEMP_DATASET_DIR > "dataset_dir.txt"
