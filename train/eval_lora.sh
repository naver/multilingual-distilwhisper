#!/bin/sh
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="gpu_32g"
#SBATCH --output=log_lora_ft_%j.log
#SBATCH --time=360:00:00



ROOT=<PATH TO DISTILWHISPER>
output_dir=<PATH HERE>
use_auth_token=<ADD HERE>


data_ids_folder="$ROOT/data/generated_ids/"
dataset_ids_file="$data_ids_folder/cv_10k.json"
seed=77

lang=$1
checkpoint=$2


student_model_name_or_path=openai/whisper-small
lora_adapter_name_or_path=$checkpoint
use_student_distilwhisper="False"
use_teacher="False"
num_train_epochs=10
warmup_ratio=0.1
use_attentions_loss="True"
dataset_ids_file="$data_ids_folder"
cpu_debug="False"
evaluate_only="True"
evaluation_output=$output_dir

echo $lang, $seed

dataset=google/fleurs

python $ROOT/model/train_lora.py --dataset $dataset \
--student_model_name_or_path $student_model_name_or_path \
--use_student_distilwhisper $use_student_distilwhisper --output_dir $output_dir \
--lang $lang --num_train_epochs $num_train_epochs --warmup_ratio $warmup_ratio \
--use_attentions_loss $use_attentions_loss --dataset_ids_file $dataset_ids_file --use_auth_token $use_auth_token \
--cpu_debug $cpu_debug --seed $seed --evaluate_only $evaluate_only --evaluation_output $evaluation_output --lora_adapter_name_or_path $lora_adapter_name_or_path


dataset=mozilla-foundation/common_voice_13_0

python $ROOT/model/train_lora.py --dataset $dataset \
--student_model_name_or_path $student_model_name_or_path \
--use_student_distilwhisper $use_student_distilwhisper --output_dir $output_dir \
--lang $lang --num_train_epochs $num_train_epochs --warmup_ratio $warmup_ratio \
--use_attentions_loss $use_attentions_loss --dataset_ids_file $dataset_ids_file --use_auth_token $use_auth_token \
--cpu_debug $cpu_debug --seed $seed --evaluate_only $evaluate_only --evaluation_output $evaluation_outpu --lora_adapter_name_or_path $lora_adapter_name_or_path






