#!/bin/bash
#SBATCH --partition=gpu_v100s
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --cpus-per-task=4        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:v100s:1             # Using 1 GPU card
#SBATCH --mem=200GB               # Request 50GB memory
#SBATCH --time=0-00:20:00        # Time limit day-hrs:min:sec
#SBATCH --output=/home/hpiao6/scratch/fme_phm.log   # Standard output
#SBATCH --error=/home/hpiao6/scratch/fme_phm.err    # Standard error log

OUTPUT=/home/hpiao6/scratch/fme_phm.out
cd /home/hpiao6/EasyEdit
'''
python Federated_ME.py --editing_method  MELO \
 --hparams_dir ./hparams/MELO/gpt2-xl.yaml \
 --data_dir "/home/hpiao6/EasyEdit/data/benchmark_ZsRE_ZsRE-test-all.json" >> /home/hpiao6/scratch/FME/melo_nlp.log 2>&1 &
'''
'''
python Federated_ME_qa.py >> /home/hpiao6/scratch/FME/melo_qa_nouse.log 2>&1 &
'''
'''
python multimodal_edit.py >> /home/hpiao6/scratch/FME/melo_vqa.log 2>&1 &
'''
'''
python Federated_ME_vqa.py >> /home/hpiao6/scratch/FME/melo_vqa_nouse.log 2>&1 &
'''
nohup python Federated_ME_qa.py >> /data2/hmpiao/FME/log/melo_qa_nouse.log 2>&1 &
wait

date        >> $OUTPUT
#