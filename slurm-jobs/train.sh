#!/bin/bash
#SBATCH --job-name=sedformer
#SBATCH --nodes=1
#SBATCH --exclude=n1,n2,n3,n4,n5,n101,n102
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sedformer/%j.stdout
#SBATCH --error=logs/sedformer/%j.stderr
#SBATCH --time=24:00:00  # Temps maximum d'exécution (hh:mm:ss)
#SBATCH --ntasks=1  # Nombre total de tâches

JOB_NAME="sedformer"

LOG_STDOUT="logs/$JOB_NAME/$SLURM_JOB_ID.stdout"
LOG_STDERR="logs/$JOB_NAME/$SLURM_JOB_ID.stderr"

function restart {
    echo "Calling restart" >> $LOG_STDOUT
    scontrol requeue $SLURM_JOB_ID
    echo "Scheduled job for restart" >> $LOG_STDOUT
}

function ignore {
    echo "Ignored SIGTERM" >> $LOG_STDOUT
}
trap restart USR1
trap ignore TERM

# Start (or restart) experiment
date >> $LOG_STDOUT
which python >> $LOG_STDOUT
echo "---Beginning program---" >> $LOG_STDOUT
echo "Exp name      : $JOB_NAME" >> $LOG_STDOUT
echo "Slurm Job ID  : $SLURM_JOB_ID" >> $LOG_STDOUT

echo "Activation de l'environnement SigLinear" >> $LOG_STDOUT
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SigLinear
echo "Environnement SigLinear activé" >> $LOG_STDOUT

conda install pytorch=1.9.0 cudatoolkit=10.2 -c pytorch

sh ~/SEDformer/scripts/run_M_Sig.sh

wait $!