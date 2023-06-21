#!/bin/bash

python3 train_ssd.py --data=data/lego --model-dir=models/lego --batch-size=16 --epochs=100 --dataset-type=voc
