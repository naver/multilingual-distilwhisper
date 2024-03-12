# Supporting functions for train.py

import json
import re
from typing import Any, Dict, List, Union
from dataclasses import dataclass

import pandas as pd
import torch
import datasets
from datasets import Audio

from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from DistilWhisper import DistilWhisperForConditionalGeneration, convert_to_distil_whisper
from dictionaries import lang_to_whisper, lang_to_dataset, idwhisper_to_lang


class ExtendedTextNormalizer(BasicTextNormalizer):
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        super(ExtendedTextNormalizer, self).__init__(remove_diacritics=remove_diacritics,split_letters=split_letters)

    def __call__(self, s: str, lang: str = None):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters or lang in ["ja", "zh", "th", "yue_HK", "lo", "my"]:
            regex = r"[\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u3100-\u312F\u3130-\u318F\u31A0-\u31BF\u31C0-\u31EF\u31F0-\u31FF\u3200-\u32FF\u3300-\u33FF\u3400-\u4DBF\u4E00-\u9FFF\uA000-\uA48F\uA490-\uA4CF\uA960-\uA97F\uAC00-\uD7AF\uF900-\uFAFF\uFE10-\uFE1F\uFF00-\uFFEF-\u0E7F\u0E80-\u0EFF\u0F00-\u0FFF\u0E00-\u0E7F\u1000-\u109F\uAA60-\uAA7F]"
            s = re.sub(regex, lambda x: " " + x.group(0) + " ", s)
            s = s.strip()

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

#metrics
def compute_metrics_transcription_lora(batch, language, processor, metrics, normalizer):

    pred_ids = batch.predictions[0] 
    label_ids = batch.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    #print("example: index 0 tokens before decoding:", label_ids[0])
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    if normalizer:
        pred_str = [normalizer(text) for text in pred_str]
        label_str = [normalizer(text) for text in label_str]
    else:
        pred_str = [processor.tokenizer._normalize(text) for text in pred_str]
        label_str = [processor.tokenizer._normalize(text) for text in label_str]

    metrics_results = {}
    wer = 100 * metrics["wer"].compute(predictions=pred_str, references=label_str)
    cer = 100 * metrics["cer"].compute(predictions=pred_str, references=label_str)
    metrics_results[f"wer_{language}"] = wer
    metrics_results[f"cer_{language}"] = cer

    return metrics_results

def compute_metrics_transcription(batch, processor, metrics, normalizer):
    pred_ids = batch.predictions
    label_ids = batch.label_ids
    
    langs = [idwhisper_to_lang[element[1]] for element in batch.predictions]

    lang_map = {}
    for i in range(len(langs)):
        if langs[i] in lang_map.keys():
            lang_map[langs[i]].append(i)
        else:
            lang_map[langs[i]] = [i]

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if normalizer:
        pred_str = [normalizer(text) for text in pred_str]
        label_str = [normalizer(text) for text in label_str]
    else:
        pred_str = [processor.tokenizer._normalize(text) for text in pred_str]
        label_str = [processor.tokenizer._normalize(text) for text in label_str]

    metrics_results = {}
    avg_wer = 0.0
    avg_cer = 0.0
    for lang in lang_map.keys():
        pred_str_lang = [pred_str[i] for i in lang_map[lang]]
        label_str_lang = [label_str[i] for i in lang_map[lang]]
        wer = 100 * metrics["wer"].compute(predictions=pred_str_lang, references=label_str_lang)
        cer = 100 * metrics["cer"].compute(predictions=pred_str_lang, references=label_str_lang)
        metrics_results["wer_"+lang] = wer
        metrics_results["cer_"+lang] = cer
        avg_wer += wer
        avg_cer += cer
    metrics_results["avg_wer"] = avg_wer/len(lang_map.keys())
    metrics_results["avg_cer"] = avg_cer/len(lang_map.keys())

    return metrics_results

def compute_metrics_translation(batch, processor, metrics, normalizer):
    pred_ids = batch.predictions
    label_ids = batch.label_ids
    langs = [idwhisper_to_lang[element[1]] for element in batch.predictions]
    lang_map = {}
    for i in range(len(langs)):
        if langs[i] in lang_map.keys():
            lang_map[langs[i]].append(i)
        else:
            lang_map[langs[i]] = [i]

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if normalizer:
        pred_str = [normalizer(text) for text in pred_str]
        label_str = [normalizer(text) for text in label_str]
    else:
        pred_str = [processor.tokenizer._normalize(text) for text in pred_str]
        label_str = [processor.tokenizer._normalize(text) for text in label_str]
    print(pred_str[0])

    metrics_results = {}
    avg_bleu = 0.0
    for lang in lang_map.keys():
        pred_str_lang = [pred_str[i] for i in lang_map[lang]]
        label_str_lang = [[label_str[i]] for i in lang_map[lang]]
        bleu = 100 * metrics["bleu"].compute(predictions=pred_str_lang, references=label_str_lang)["bleu"]
        metrics_results["bleu_"+lang] = bleu
        avg_bleu += bleu
    metrics_results["avg_bleu"] = avg_bleu/len(lang_map.keys())

    return metrics_results

#helpers
def freeze_original_whisper_parameters(model):
    for p in model.parameters():
        p.requires_grad = False  # freezing
    for name, p in model.named_parameters():
        if ("ffn_clsr" in name) and (not "shared" in name):
            p.requires_grad = True

def load_teacher_model(args, device):
    ### Load teacher model classes ###
    if args.use_teacher: 
        # Build Whisper architecture + load weights
        if args.use_teacher_distilwhisper:
            return DistilWhisperForConditionalGeneration.from_pretrained(args.teacher_model_name_or_path).to(device)
        else:
            if args.load_in_8bit:
                print("Loading model in LLM.int8")
                return WhisperForConditionalGeneration.from_pretrained(args.teacher_model_name_or_path,
                                                                        device_map="sequential",
                                                                        load_in_8bit=args.load_in_8bit)
            elif args.load_in_4bit:
                print("Loading model in LLM.int8")
                return WhisperForConditionalGeneration.from_pretrained(args.teacher_model_name_or_path,
                                                                        device_map="sequential",
                                                                        load_in_4bit=args.load_in_4bit)
            else:
                return WhisperForConditionalGeneration.from_pretrained(args.teacher_model_name_or_path).to(device)
    return None

def load_student_model(args, processor, device):
    ### Load student model ###
    if args.use_student_distilwhisper:
        student_model = DistilWhisperForConditionalGeneration.from_pretrained(args.student_model_name_or_path).to(device)
        #inference of full ft
        if args.train_whisper:
            student_model.config.skip_gate_prob=1.0
    else:
        student_model = WhisperForConditionalGeneration.from_pretrained(args.student_model_name_or_path).to(device)

        if not args.train_whisper:
            # Convert Whisper to DistilWhisper #
            student_model = convert_to_distil_whisper(student_model, processor=processor, clsr_langs=[lang_to_whisper[args.lang].lower()], skip_gate_prob=0.2, use_gate_budget=False, gate_budget=0.0, device=device)
            student_model = student_model.to(device)
        else:
            student_model = convert_to_distil_whisper(student_model, processor=processor, clsr_langs=[lang_to_whisper[args.lang].lower()], skip_gate_prob=1.0, use_gate_budget=False, gate_budget=0.0, device=device)
            student_model = student_model.to(device)
    
    # If should restart the gates
    if args.restart_gates:
        student_model.restart_gates()
    # Solution to make gradient checkpointing work with network freeze
    # From: https://github.com/huggingface/transformers/issues/23170#issuecomment-1536455122
    """
    if hasattr(student_model, "enable_input_require_grads"):
        student_model.enable_input_require_grads()
        print("enable_input_require_grads -------------------")
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        student_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    """
    # Freeze original parameters of student model #
    if not args.train_whisper:
        freeze_original_whisper_parameters(student_model)

    return student_model

#dataset loaders
def prepare_dataset(batch, processor):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = \
    processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

def print_dataset_stats(args):
    import numpy as np
    lang_dataset = lang_dataset.map(lambda example: {"tokens_number": len(example["labels"])})
    lang_dataset = lang_dataset.map(lambda example: {"words_number": len(example["transcription"].split())})
    lang_dataset = lang_dataset.map(lambda example: {"audio_len": len(example["audio"]["array"])/16000.0})
    print(f"Lang {args.lang} stats:")
    token_avg = np.average(lang_dataset["train"]["tokens_number"])
    words_avg = np.average(lang_dataset["train"]["words_number"])
    audio_avg = np.average(lang_dataset["train"]["audio_len"])
    print(f"Train - tokens avg = {token_avg} words avg = {words_avg} audio len avg = {audio_avg}")
    token_avg = np.average(lang_dataset["validation"]["tokens_number"])
    words_avg = np.average(lang_dataset["validation"]["words_number"])
    audio_avg = np.average(lang_dataset["validation"]["audio_len"])
    print(f"Validation - tokens avg = {token_avg} words avg = {words_avg} audio len avg = {audio_avg}")
    token_avg = np.average(lang_dataset["test"]["tokens_number"])
    words_avg = np.average(lang_dataset["test"]["words_number"])
    audio_avg = np.average(lang_dataset["test"]["audio_len"])
    print(f"Test - tokens avg = {token_avg} words avg = {words_avg} audio len avg = {audio_avg}")

def load_dataset(args, dataset_name, evaluate_only=False):
    '''
    Loads a HF dataset for training or evaluation 
    '''
    lang = args.lang
    if lang in lang_to_dataset[dataset_name]:
        #loads from HF or cache
        print(f"Loading dataset {dataset_name} for {lang}")
        splits = ["validation", "test"] if evaluate_only else ["train", "validation", "test"]
        lang_dataset = datasets.load_dataset(dataset_name, lang_to_dataset[dataset_name][lang], "+".join(splits),
                                            streaming=False,
                                            use_auth_token=args.use_auth_token if args.use_auth_token else False)
        # Add the column for language
        for split in splits: 
            lang_dataset[split] = lang_dataset[split].add_column("lang", len(lang_dataset[split])*[lang])
        # Subsample if it is just a code test
        if args.cpu_debug:
            lang_dataset["train"] = lang_dataset["train"].select(list(range(10)))
            lang_dataset["validation"] = lang_dataset["validation"].select(list(range(10)))
            lang_dataset["test"] = lang_dataset["test"].select(list(range(10)))
        
        # Correct formats if it is common voice:
        if dataset_name == "mozilla-foundation/common_voice_13_0":
            # Select ids from CV for training ONLY
            if args.dataset_ids_file and not evaluate_only and not args.cpu_debug:
                # Open File
                print("Selecting ids of the dataset")
                dataset_ids = json.load(open(args.dataset_ids_file))
                dataset_ids = dataset_ids[lang_to_dataset[dataset_name][lang]]
                # TO DO - assert is a list of ints
                lang_dataset["train"] = lang_dataset["train"].select(dataset_ids["train"])
                lang_dataset["validation"] = lang_dataset["validation"].select(dataset_ids["validation"])
            del lang_dataset["other"]
            del lang_dataset["invalidated"]
            lang_dataset = lang_dataset.rename_column("sentence", "transcription")
            lang_dataset = lang_dataset.cast_column("audio", Audio(sampling_rate=16000))

        if evaluate_only and "train" in lang_dataset:
            #BUG: cached CV is loading train split even when requested is "validation+test"
            del lang_dataset["train"]
        
        return lang_dataset
    else:
        print(f"Lang {lang} not present in the dataset {dataset_name}")
        exit(1)

def load_st_dataset(args, dataset_name):
    '''
    This function works for google/fleurs only
    '''
    lang = args.lang
    #loads source in args.lang
    lang_dataset = load_dataset(args, dataset_name, evaluate_only=True)
    
    #loads target
    splits = ["train", "validation", "test"]
    dataset_translation_ids = json.load(open(args.dataset_ids_file))
    dataset_en = datasets.load_dataset(dataset_name, "en_us", "+".join(splits), streaming=False)
    dataset_en = datasets.concatenate_datasets([dataset_en[split] for split in splits])
    dataset_en = dataset_en.add_column("idx", list(range(len(dataset_en))))
    dataset_en = dataset_en.remove_columns("audio")
    lang_ids = dataset_translation_ids[lang] #lang_to_dataset[dataset_name][lang]]
    dataset_en_lang = dataset_en.select(lang_ids)

    #matches source and target for test only
    lang_dataset["test"] = lang_dataset["test"].filter(lambda example: example["id"] in dataset_en_lang["id"])
    translations = []
    dataset_en_lang = pd.DataFrame(dataset_en_lang)
    for i in lang_dataset["test"]["id"]:
        translations.append(dataset_en_lang.loc[dataset_en_lang["id"] == i]["transcription"].to_list()[0])
    assert len(lang_dataset["test"]) == len(translations)

    #ugly fix to use prepare_dataset from ASR
    lang_dataset["test"] = lang_dataset["test"].remove_columns("transcription").add_column("transcription", translations)
    return lang_dataset

#evaluation
def run_asr_evaluation(distilwhisper, dataset):
    asr_evaluation_dict = dict()
    for split in ["validation", "test"]:
        print("Performance on", split)
        asr_evaluation_dict[split] = distilwhisper.predict(test_dataset=dataset[split]).metrics
        print(asr_evaluation_dict[split])
    return asr_evaluation_dict

# logging 
def write_json(json_file_name, json_data, mode="w"):
    with open(json_file_name, mode=mode, encoding='utf-8') as output_file:
        json.dump(json_data, output_file, ensure_ascii=False, indent=2, separators=(',', ': '))

def write_clear_log(tsv_file_name, json_data, lang, metric):
    header = json_data.keys()
    scores = list()
    for split in header:
        scores.append(json_data[split]["test_"+ metric +"_" + lang])

    print("Writing output scores at", tsv_file_name)
    with open(tsv_file_name, "w") as output_file:
        output_file.write("\t".join(header)+ "\n")
        output_file.write("\t".join([str(score) for score in scores]) + "\n")

def write_inference_scores(dictionary, output_dir, lang, dataset_name, task="asr"):
    dataset_dict = {"google/fleurs": "FLEURS", "mozilla-foundation/common_voice_13_0": "CV"}
    file_prefix = output_dir + "/evaluation_" + task + "_" + lang + "_" + dataset_dict[dataset_name]
    #dump all stats on a json file
    write_json(file_prefix + ".json", dictionary)
    metric = "wer" if task == "asr" else "bleu" 
    write_clear_log(file_prefix + ".tsv", dictionary, lang, metric=metric)

