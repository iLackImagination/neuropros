#!/bin/bash -i
#source /Home/miniconda3/etc/profile.d/conda.sh
#conda init bash
conda activate phd

python3 ./training_optim.py gamma1 none
python3 ./training_optim.py alpha1 none
python3 ./training_optim.py rho1 none
python3 ./training_optim.py gamma2 none
python3 ./training_optim.py alpha1 alpha2
