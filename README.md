# Multilingual DistilWhisper: Efficient Distillation of Multi-task Speech Models via Language-Specific Experts

Multilingual Distilwhisper allows for better ASR performance in target languages by adding lightweight CLSR modules on top of whisper-small.

The pre-trained weights for the [paper](https://arxiv.org/abs/2311.01070) experiments are available [here](https://huggingface.co/collections/naver/multilingual-distilwhisper-6576ecae8d209fc6a767d9e7).


## Requirements

For training DistilWhisper, please install the requirements listed in *full_requirements.yaml*

For training the LoRA baseline, please install the requirements listed in *lora_requirements.yaml*

## How to run training

Scripts are available at DistilWhisper/train/
* LoRA baseline: DistilWhisper/train/train_lora.sh
* Model using only CLSR trained on ASR loss (no distillation): DistilWhisper/train/train_clsr.sh
* Distillation model with JS loss (DistilWhisper): DistilWhisper/train/train_js_distill.sh
* Distillation model with KL loss (From the appendix available on arxiv): DistilWhisper/train/train_kl_distill.sh

## How to run evaluation

Scripts are available at DistilWhisper/train/
* LoRA baseline: DistilWhisper/train/eval_lora.sh
* Model using only CLSR trained on ASR loss (no distillation): DistilWhisper/train/eval_clsr.sh
* Distillation model with JS loss (DistilWhisper): DistilWhisper/train/eval_js_distill.sh
* Distillation model with KL loss (From the appendix): DistilWhisper/train/eval_kl_distill.sh

## Interative inference example

Check example at model/inference_example.py

```
from transformers import WhisperProcessor
from DistilWhisper import DistilWhisperForConditionalGeneration
from datasets import Audio, load_dataset

# 1. load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = DistilWhisperForConditionalGeneration.from_pretrained("naver/multilingual-distilwhisper-28k")

# 2. load language experts 
language="calatan"
model.load_experts(language)
model.set_language_task(language=language, task="transcribe")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

# 3. load streaming dataset and read first audio sample
ds = load_dataset("google/fleurs", "ca_es", split="test")
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
input_speech = next(iter(ds))["audio"]
input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

# 4. generate token ids and decode it into text
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
# ["Els esports més coneguts són el futbol, el bàsquet, el voleibol, el waterpolo, l'esgrima, el rucbi, el ciclisme, hoquei sobre gel, hoquei sobre patins i l'automobilisme de Fórmula U."]
```

## Citation
```
@inproceedings{ferraz2024distilwhisper,
  title={Multilingual DistilWhisper: Efficient Distillation of Multi-task Speech Models via Language-Specific Experts},
  author={Ferraz, Thomas Palmeira and Boito, Marcely Zanon and Brun, Caroline and Nikoulina, Vassilina},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```
