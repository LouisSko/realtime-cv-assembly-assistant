#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu_8
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=finetuning_ssd
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ulhni@student.kit.edu
#SBATCH --output=/home/kit/stud/ulhni/aiss/object-detection-project/pytorch-ssd/training_results/output.txt
#SBATCH --error=/home/kit/stud/ulhni/aiss/object-detection-project/pytorch-ssd/training_results/error.txt

# Activate env
source /home/kit/stud/ulhni/aiss_env/bin/activate

# navigate into patorch-ssd directory
cd /home/kit/stud/ulhni/aiss/object-detection-project/pytorch-ssd
#$ wget https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O models/mobilenet-v1-ssd-mp-0_675.pth
#$ pip3 install -v -r requirements.txt


# Run Python script
# python3 train_ssd.py --data=data/lego_train_structure --model-dir=models/lego --net mb1-ssd --resolution=600 --batch-size=32 --epochs=100 --dataset-type=voc

python3 train_ssd.py --data=data/lego_data_test --model-dir=models/lego --net=mb1-ssd  --pretrained-ssd=models/mobilenet-v1-ssd-mp-0_675.pth --batch-size=32 --epochs=100 --dataset-type=voc

python3 train_ssd.py --data=data/lego_data_test --model-dir=models/lego --net=mb1-ssd --batch-size=16 --epochs=100 --dataset-type=voc



python3 eval_ssd.py --net=mb1-ssd --dataset=data/lego_data_test --model=models/lego/mb1-ssd-Epoch-10-Loss-3.2922821044921875.pth --label_file=models/lego/labels.txt


python3 onnx_export.py --net=mb1-ssd --input=models/lego/mb1-ssd-Epoch-10-Loss-3.2922821044921875.pth --output=models/ --labels=models/lego/labels.txt 


# make executable: chmod +x /home/kit/stud/ulhni/aiss/object-detection-project/job_train_ssdv1.sh
# run: sbatch /home/kit/stud/ulhni/aiss/object-detection-project/job_train_ssdv1.sh