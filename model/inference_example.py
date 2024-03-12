# Based on the example provided by OpenAI: https://huggingface.co/openai/whisper-large

from transformers import WhisperProcessor
from DistilWhisper import DistilWhisperForConditionalGeneration
from datasets import Audio, load_dataset

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = DistilWhisperForConditionalGeneration.from_pretrained("naver/multilingual-distilwhisper-28k")

# load catalan experts and set language
model.load_experts("catalan")
model.set_language_task(language="catalan", task="transcribe")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="catalan", task="transcribe")

# load streaming dataset and read first audio sample
ds = load_dataset("google/fleurs", "ca_es", split="test")
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
input_speech = next(iter(ds))["audio"]
input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids)
print(transcription)
# ["<|startoftranscript|><|ca|><|transcribe|><|notimestamps|>Els esports més coneguts són el futbol, el bàsquet, el voleibol, el waterpolo, l'esgrima, el rucbi, el ciclisme, hoquei sobre gel, hoquei sobre patins i l'automobilisme de Fórmula U.<|endoftext|>"]

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
# ["Els esports més coneguts són el futbol, el bàsquet, el voleibol, el waterpolo, l'esgrima, el rucbi, el ciclisme, hoquei sobre gel, hoquei sobre patins i l'automobilisme de Fórmula U."]