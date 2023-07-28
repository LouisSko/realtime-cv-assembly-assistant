#!/bin/bash

python3 train_ssd.py --data=data/lego_data_test --model-dir=models/lego --net mb1-ssd --base-net models/mobilenet-v1-ssd-mp-0.675.pth --resolution=300 --batch-size=32 --epochs=100 --dataset-type=voc

#python3 train_ssd.py --data=data/lego_train_structure --model-dir=models/lego --net mb2-ssd-lite --base-net models/mobilenet-v1-ssd-mp-0.675.pth --resolution=600 --batch-size=32 --epochs=100 --dataset-type=voc

python3 train_ssd.py --data=data/lego_data_test --model-dir=models/lego --net vgg16-ssd --base-net models/vgg16_reducedfc.pth --resolution=600 --batch-size=32 --epochs=100 --dataset-type=voc

