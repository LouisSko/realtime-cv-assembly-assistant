#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu_8
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=finetuning_ssd
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ulhni@student.kit.edu
#SBATCH --output=/home/kit/stud/ulhni/aiss/object-detection-project/pytorch-ssd/training_results/output_vgg.txt
#SBATCH --error=/home/kit/stud/ulhni/aiss/object-detection-project/pytorch-ssd/training_results/error_vgg.txt

# Activate env
source /home/kit/stud/ulhni/aiss_env/bin/activate

# navigate into patorch-ssd directory
cd /home/kit/stud/ulhni/aiss/object-detection-project/pytorch-ssd
#$ wget https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O models/mobilenet-v1-ssd-mp-0_675.pth
#$ pip3 install -v -r requirements.txt


# Run Python script
python3 train_ssd.py --data=data/lego_train_structure --model-dir=models/lego --net vgg16-ssd --base-net models/vgg16_reducedfc.pth --batch-size=32 --epochs=100 --dataset-type=voc --lr=0.0001





# eval performance
# python3 eval_ssd.py --net=vgg16-ssd --model=models/lego/vgg16-ssd-Epoch-22-Loss-5.372615814208984.pth --dataset=data/lego_train_structure/

# make executable: chmod +x /home/kit/stud/ulhni/aiss/object-detection-project/job_train_vgg.sh
# run: sbatch /home/kit/stud/ulhni/aiss/object-detection-project/job_train_vgg.sh
