# Code for distilling whisper

import argparse
from distutils.util import strtobool
import gc
import os
import time
from functools import partial

import torch
import evaluate
from transformers import WhisperProcessor
from DistillationTrainer import DistillationTrainer, DistillationTrainingArguments
from torch.nn.parallel.data_parallel import DataParallel

from train_utils import *

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
        "--teacher_model_name_or_path", type=str, help="Model name", default="openai/whisper-large-v2"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, help="Model name", default="openai/whisper-large-v2"
    )
    parser.add_argument(
        "--use_teacher", type=lambda x: bool(strtobool(x)), help="Distillation setting", default=False
    )
    parser.add_argument(
        "--use_teacher_distilwhisper", type=lambda x: bool(strtobool(x)), help="Is teacher a DistilWhisper", default=False
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
        "--temperature", type=float, help="Distillation temperature", default=1.0
    )
    parser.add_argument( #TO DO: FIX THIS
        "--sigma", 
        nargs="*",  # 0 or more values expected => creates a list
        type=float,
        default=[1.0, 3.1416],
    )
    parser.add_argument(
        "--fp16", type=lambda x: bool(strtobool(x)), help="fp16", default=False
    )
    parser.add_argument(
        "--num_proc", type=int, help="Number of CPU processes for data pre-processing", default=2
    )

    parser.add_argument(
        "--evaluation_output", type=str, help="Folder por storing the scoring output", default="./"
    )
    parser.add_argument(
        "--use_gate_loss", type=lambda x: bool(strtobool(x)), help="Use gate budget loss", default=False
    )
    parser.add_argument(
        "--gate_budget", type=float, help="Ratio of gates to be activated", default=0.5
    )
    parser.add_argument(
        "--use_attentions_loss", type=lambda x: bool(strtobool(x)), help="", default=False
    )
    parser.add_argument(
        "--loss_type", type=str, help="Knowledge distillation loss type: JS or KL", default="JS5"
    )
    parser.add_argument(
        "--restart_gates", type=lambda x: bool(strtobool(x)), help="Gate restart before training", default=False
    )
    parser.add_argument(
        "--evaluate_translation", type=lambda x: bool(strtobool(x)), help="Cross-task evaluation", default=False
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

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        langs = [feature["lang"] for feature in features]
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

        batch["lang"] = langs

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
    print(args)
    gc.collect()
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create processor, wrapper of feature_extractor and tokenizer
    task = "translate" if args.evaluate_translation else "transcribe"
    print("\tPREPARING TO:", task)
    processor = WhisperProcessor.from_pretrained(args.tokenizer_name_or_path, task=task)

    ## Set up functions for evaluation 
    metrics = {}
    if args.evaluate_translation:
        metrics["bleu"] = evaluate.load("bleu")
    metrics["wer"] = evaluate.load("wer")
    metrics["cer"] = evaluate.load("cer")

    # Tokenizer output format
    #dtype = torch.half if (args.load_in_8bit or args.load_in_4bit) else torch.float
    # TO DO: check if new versions of Transformers solve this issue

    # Workaround for multilingual normalization - TO DO: remove when transformers fix
    normalizer = ExtendedTextNormalizer(split_letters=True)

    # Pre-processing func
    prepare_dataset_func = partial(prepare_dataset, processor=processor)

    # Prediction func
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Compute Metrics func
    task_metrics = compute_metrics_translation if args.evaluate_translation else compute_metrics_transcription
    compute_metrics_fn = partial(task_metrics, 
                            processor=processor, 
                            metrics=metrics,
                            normalizer=normalizer)

    # Data loading
    if args.evaluate_translation:
        print("\tLoading ST dataset", args.dataset)
        lang_dataset = load_st_dataset(args, args.dataset)
        dataset = lang_dataset.map(prepare_dataset_func, num_proc=args.num_proc)
    else:
        lang_dataset = load_dataset(args, args.dataset, evaluate_only=args.evaluate_only)
        if args.print_dataset_stat:
            print("\tLoading ASR dataset", args.dataset)
            print_dataset_stats(dataset)
        dataset = lang_dataset.map(prepare_dataset_func, num_proc=args.num_proc)
    
    if os.path.exists(args.output_dir):
        args.output_dir = args.output_dir+"-"+str(time.time())
    
    # Set up teacher (if use_teacher) and student models
    teacher_model = load_teacher_model(args, device)
    student_model = load_student_model(args, processor, device)
    
    ### Define training args ###
    training_args = DistillationTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
        use_mps_device=False,
        torch_compile=False,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        adam_beta2=args.adam_beta2,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        predict_with_generate=True,
        generation_max_length=225,
        # logging & evaluation strategies
        #logging_dir=f"{repo_name}/logs",
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
        metric_for_best_model="avg_wer",
        report_to="tensorboard",
        # push to hub parameters
        #push_to_hub=True,
        #hub_strategy="every_save",
        #hub_model_id=repo_name,
        #hub_token=HfFolder.get_token(),
        # distilation parameters
        temperature=args.temperature,
        use_teacher=args.use_teacher,
        use_kd_warm_up_scheduling=args.use_kd_warm_up_scheduling,
        learnable_loss_weight=False,
        use_gate_loss=args.use_gate_loss,
        gate_budget=args.gate_budget,
        use_attentions_loss=args.use_attentions_loss,
        loss_type=args.loss_type,
        translation_task=args.evaluate_translation,
        train_whisper=args.train_whisper
    )

    if args.apply_spec_augment:
        student_model.config.apply_spec_augment = student_model.model.config.apply_spec_augment = True
        student_model.config.mask_feature_prob = student_model.model.config.mask_feature_prob = 0.1

    ### Load Distiller and possibly train it ###
    distilwhisper = DistillationTrainer(
        student_model,
        training_args,
        teacher_model=teacher_model,
        train_dataset=dataset["train"] if not args.evaluate_only else None,
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        group_ids_column="lang",
        sigma=args.sigma,
        compute_metrics=compute_metrics_fn, 
    )

    if not args.evaluate_only:
        gc.collect()
        torch.cuda.empty_cache()
        distilwhisper.train(resume_from_checkpoint=args.resume_from_checkpoint)
        distilwhisper.save_model(args.output_dir)
    

    # Evaluation
    if args.evaluate_only:
        print("\tEVALUATION:")
        if args.evaluate_translation:
            print("\tNumber of utterances:", len(dataset["test"]))
            score_metrics = dict()
            score_metrics["test"] = distilwhisper.predict(test_dataset=dataset["test"]).metrics 
            write_inference_scores(score_metrics, args.evaluation_output, args.lang, args.dataset, task="st")
        else:
            in_dataset = dataset
            #retrieve wer metrics
            score_metrics = run_asr_evaluation(distilwhisper, in_dataset)
            #send to write function
            write_inference_scores(score_metrics, args.evaluation_output, args.lang, args.dataset)

        return
    
if __name__ == '__main__':
    main()
