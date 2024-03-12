#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="gpu_80g"
#SBATCH --output=eval_js_temp1_%j.log
#SBATCH --time=360:00:00


ROOT=<PATH TO DISTILWHISPER>
output_dir=<PATH HERE>
use_auth_token=<ADD HERE>


data_ids_folder="$ROOT/data/generated_ids/"
dataset_ids_file="$data_ids_folder/cv_10k.json"
lang=$1
checkpoint=$2

student_model_name_or_path=$checkpoint
use_student_distilwhisper="True" #loads a distilwhisper model
use_teacher="True" 
temperature=1.0
sigma="1.41 1.0"
use_gate_loss="True"
num_train_epochs=10
warmup_ratio=0.1
use_attentions_loss="True"
loss_type="KL"
evaluate_only="True"
cpu_debug="False"

dataset=mozilla-foundation/common_voice_13_0

python $ROOT/model/train_distillwhisper.py --dataset $dataset \
--student_model_name_or_path $student_model_name_or_path --use_teacher $use_teacher --use_student_distilwhisper $use_student_distilwhisper \
--output_dir $output_dir --temperature $temperature --sigma $sigma --use_gate_loss $use_gate_loss \
--num_train_epochs $num_train_epochs --warmup_ratio $warmup_ratio --use_attentions_loss $use_attentions_loss --loss_type $loss_type \
--dataset_ids_file $dataset_ids_file --use_auth_token $use_auth_token --cpu_debug $cpu_debug --evaluation_output $output_dir \
--seed $seed --evaluate_only $evaluate_only


dataset=google/fleurs

python $ROOT/model/train_distillwhisper.py --dataset $dataset \
--student_model_name_or_path $student_model_name_or_path --use_teacher $use_teacher --use_student_distilwhisper $use_student_distilwhisper \
--output_dir $output_dir --temperature $temperature --sigma $sigma --use_gate_loss $use_gate_loss \
--num_train_epochs $num_train_epochs --warmup_ratio $warmup_ratio --use_attentions_loss $use_attentions_loss --loss_type $loss_type \
--dataset_ids_file $dataset_ids_file --use_auth_token $use_auth_token --cpu_debug $cpu_debug --evaluation_output $output_dir \
--seed $seed --evaluate_only $evaluate_only
