#!/bin/bash
#SBATCH -J photXclus
#SBATCH --ntasks 50
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mail-user=john.doe@example.com
#SBATCH --mail-type=END
#SBATCH -e photXclus-%j.err
#SBATCH -o photXclus-%j.out

#The file should be launch as follows
#sbatch slurm_example.sh 10 
#this will run the code on 50 clusters starting from the cluster number 10  

export OMP_NUM_THREADS=1

cinf=`echo $1 | awk '{ printf "%i\n", $1}'`
csup=`echo $1 | awk '{ printf "%i\n", $1+49}'`

source ~/.bashrc
conda activate photXclus
for (( iclus=$cinf; iclus<=$csup; iclus++ ))
do
    echo $iclus
    srun -N 1 -n 1 python run_oneclus.py -i $iclus -inf $cinf -p params.py >& /home/jdoe/photXclus/logs/log_photXclus_${iclus}.log & 
done
wait
