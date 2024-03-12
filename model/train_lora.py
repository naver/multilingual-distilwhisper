# Code for distilling whisper

import argparse
from distutils.util import strtobool
import gc
import os
import time
from functools import partial

import torch
import evaluate
from transformers import WhisperProcessor, Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.nn.parallel.data_parallel import DataParallel

import adapters

from train_utils import *
from peft import LoraConfig, PeftModel, LoraConfig, get_peft_model, PeftConfig

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, help="Seed", default=42
    )
    parser.add_argument(
        "--student_model_name_or_path", type=str, help="Model name", default="openai/whisper-small"
    )
    parser.add_argument(
        "--lora_adapter_name_or_path", type=str, help="Adapter name", default=None
    )
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, help="Model name", default="openai/whisper-large-v2"
    )
    parser.add_argument(
        "--use_student_distilwhisper", type=lambda x: bool(strtobool(x)), help="Is student a DistilWhisper", default=False
    )
    parser.add_argument(
        "--use_kd_warm_up_scheduling", type=lambda x: bool(strtobool(x)), help="use_kd_warm_up_scheduling", default=False
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset name", default="google/fleurs"
    )
    parser.add_argument(
        "--cpu_debug", type=lambda x: bool(strtobool(x)), help="CPU debugging test", default=False
    )
    parser.add_argument(
        "--restart_gates", type=lambda x: bool(strtobool(x)), help="Gate restart before training", default=False
    )
    parser.add_argument(
        "--evaluate_only", type=lambda x: bool(strtobool(x)), help="Evaluate only", default=False
    )
    parser.add_argument(
        "--train_whisper", type=lambda x: bool(strtobool(x)), help="Fine-tuning setting", default=False
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, help="Checkpoint file to resume from", default=None
    )
    parser.add_argument(
        "--lang", type=str, help="Language key", default=None,
    )
    parser.add_argument(
        "--load_in_8bit", type=lambda x: bool(strtobool(x)), help="LLM int8 quantization", default=False
    )
    parser.add_argument(
        "--load_in_4bit", type=lambda x: bool(strtobool(x)), help="LLM int4 quantization", default=False
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory", default="./test"
    )
    parser.add_argument(
        "--dataset_ids_file", type=str, help="Dataset ids for filtering (CV-3k/10k/28k setting)", default=None
    )
    parser.add_argument(
        "--use_auth_token", type=str, help="HF token for private datasets", default=""
    )
    parser.add_argument(
        "--num_train_epochs", type=int, help="Number of Training Epochs", default=20
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, help="Training Batch Size", default=8
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, help="Evaluation Batch Size", default=1
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, help="Gradient accumulation in number of steps", default=2
    )
    parser.add_argument(
        "--learning_rate", type=float, help="", default=1e-4
    )
    parser.add_argument(
        "--weight_decay", type=float, help="", default=0.0
    )
    parser.add_argument(
        "--adam_beta2", type=float, help="", default=0.98
    )
    parser.add_argument(
        "--warmup_ratio", type=float, help="Percentage of the traning set as warm up", default=0.05
    )
    parser.add_argument(
        "--save_steps", type=float, help="", default=None
    )
    parser.add_argument(
        "--fp16", type=lambda x: bool(strtobool(x)), help="fp16", default=False
    )
    parser.add_argument(
        "--num_proc", type=int, help="Number of CPU processes for data pre-processing", default=2
    )
    parser.add_argument(
        "--lora_r", type=int, help="Lora R dimension", default=64
    )
    parser.add_argument(
        "--lora_alpha", type=int, help="Lora R dimension", default=64
    )
    parser.add_argument(
        "--lora_dropout", type=int, help="Lora R dimension", default=0.05
    )
    parser.add_argument(
        "--load_peft", type=str, help="Model name", default=None
    )
    parser.add_argument(
        "--evaluation_output", type=str, help="Folder por storing the scoring output", default="./"
    )
    parser.add_argument(
        "--use_attentions_loss", type=lambda x: bool(strtobool(x)), help="", default=False
    )
    parser.add_argument(
        "--loss_type", type=str, help="Knowledge distillation loss type: JS or KL", default="JS5"
    )
    parser.add_argument(
        "--apply_spec_augment", type=lambda x: bool(strtobool(x)), help="Apply SpecAugment", default=False
    )
    parser.add_argument(
        "--print_dataset_stat", type=lambda x: bool(strtobool(x)), help="Print dataset stats once loaded", default=False
    )
    parser.add_argument(
        "--mask_feature_prob", type=float, help="Mask prob on SpecAugment", default=0.1
    )
    parser.add_argument(
        "--transcription_column_map", type=str,
        help="Which column represent the transcription, default assumes is 'transcription'.", default=None
    )
    # TO DO - Help comments in the arguments
    args = parser.parse_args()
    return args

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

    def set_language_task(self, model, lang, task="transcribe", step_proportion=None, is_warm_up=None):
        if isinstance(model, DataParallel):
            model.module.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=lang_to_whisper[lang],
                                                                                           task=task)
        else:
            model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=lang_to_whisper[lang],
                                                                                    task=task)
        if isinstance(model, DistilWhisperForConditionalGeneration):
            model.set_language_task(language=lang_to_whisper[lang], step_proportion=step_proportion, is_warm_up=is_warm_up, task=task)

def main():
    args = arg_parse()
    gc.collect()
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create processor, wrapper of feature_extractor and tokenizer
    task = "transcribe"
    print("\tPREPARING TO:", task)
    processor = WhisperProcessor.from_pretrained(args.tokenizer_name_or_path, task=task)

    ## Set up functions for evaluation 
    metrics = {}
    metrics["wer"] = evaluate.load("wer")
    metrics["cer"] = evaluate.load("cer")

    # Tokenizer output format

    # Workaround for multilingual normalization - TO DO: remove when transformers fix
    normalizer = ExtendedTextNormalizer(split_letters=True)

    # Pre-processing func
    prepare_dataset_func = partial(prepare_dataset, processor=processor)

    # Prediction func
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Compute Metrics func
    lang = args.lang # < it needs to be transformed into a whisper id
    task_metrics = compute_metrics_transcription_lora
    compute_metrics_fn = partial(task_metrics,
                            language=lang, 
                            processor=processor, 
                            metrics=metrics,
                            normalizer=normalizer)

    # Data loading
    lang_dataset = load_dataset(args, args.dataset, evaluate_only=args.evaluate_only)
    if args.print_dataset_stat:
        print("\tLoading ASR dataset", args.dataset)
        print_dataset_stats(dataset)
    dataset = lang_dataset.map(prepare_dataset_func, num_proc=args.num_proc)
    
    if os.path.exists(args.output_dir):
        args.output_dir = args.output_dir+"-"+str(time.time())
    
    if not args.evaluate_only:
        student_model = load_student_model(args, processor, device)
        config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["fc1", "fc2"], lora_dropout=args.lora_dropout, bias="none")
        student_model = get_peft_model(student_model, config)
        student_model.print_trainable_parameters()

    else:
        student_model = load_student_model(args, processor, device)
        student_model.load_adapter(args.lora_adapter_name_or_path)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        predict_with_generate=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        eval_accumulation_steps=30,
        generation_max_length=255,
        logging_strategy="steps",  # to get more information to TB - TO DO: make more frequent
        logging_first_step=True,
        logging_steps=1/(4*args.num_train_epochs), # Log 4 times per epoch
        evaluation_strategy="steps" if args.save_steps else "epoch",
        eval_steps=args.save_steps,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="loss",
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
    )

    lora_model = Seq2SeqTrainer(
        args=training_args,
        model=student_model,
        train_dataset=dataset["train"].remove_columns("lang") if not args.evaluate_only else None,
        eval_dataset=dataset["validation"].remove_columns("lang"),
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
    )
    
    if not args.evaluate_only:
        gc.collect()
        torch.cuda.empty_cache()
        lora_model.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    else:
        in_dataset = dataset
        #retrieve wer metrics
        score_metrics = run_asr_evaluation(lora_model, in_dataset)
        #send to write function
        write_inference_scores(score_metrics, args.evaluation_output, args.lang, args.dataset)


if __name__ == '__main__':
    main()
