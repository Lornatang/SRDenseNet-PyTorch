import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/ImageNet/original --output_dir ../data/ImageNet/SRDenseNet/train --image_size 100 --step 100 --num_workers 10")

# Split train and valid
os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/ImageNet/SRDenseNet/train --valid_images_dir ../data/ImageNet/SRDenseNet/valid --valid_samples_ratio 0.1")
