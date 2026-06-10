# Instalación de dependencias necesarias en Google Colab
!pip install -q transformers peft bitsandbytes datasets accelerate

import os
import re

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import PeftModel
from google.colab import drive


# Fijación de semilla para facilitar la reproducibilidad.
set_seed(42)

print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)

drive.mount("/content/drive", force_remount=True)

# Rutas principales del proyecto en Google Drive.
BASE_PATH = "/content/drive/MyDrive/TFG ULTIMO"
MODELS_PATH = f"{BASE_PATH}/models"
DATA_REF_PATH = f"{BASE_PATH}/data_reference"
GENERATED_PATH = f"{BASE_PATH}/generated_texts"

os.makedirs(MODELS_PATH, exist_ok=True)



# Textos seleccionados en la iteración anterior.
# En este caso, se usan los textos de iter4 para entrenar iter5.
SELECTED_JSONL = f"{GENERATED_PATH}/selected_iter4_for_iter5.jsonl"

ds = load_dataset("json", data_files=SELECTED_JSONL, split="train")

print("Total ejemplos:", len(ds))

print("Contienen @xmath:",
      sum(bool(re.search(r"@xmath", x["text"])) for x in ds))

print("\nPrimer ejemplo:\n")
print(ds[0]["text"])



SELECTED_JSONL = f"{GENERATED_PATH}/selected_iter4_for_iter5.jsonl"
selected_ds = load_dataset("json", data_files=SELECTED_JSONL, split="train")

print("Seleccionados:", len(selected_ds))

HUMAN_TRAIN_JSON = f"{DATA_REF_PATH}/train_split_iter0.json"
human_train = load_dataset("json", data_files=HUMAN_TRAIN_JSON, split="train")
human_train = human_train.map(
    lambda x: {
        "source": "human",
        "label": 0
    }
)

print("Human train original:", len(human_train))

n_sel = len(selected_ds)

MIXED_OUT_JSONL = f"{GENERATED_PATH}/mixed_iter4_for_iter5.jsonl"

n_human = min(int(n_sel * 2), len(human_train))

human_sample = human_train.shuffle(seed=42).select(range(n_human))

mixed = concatenate_datasets([selected_ds, human_sample]).shuffle(seed=42)

print("Selected:", len(selected_ds))
print("Human sample añadido:", len(human_sample))
print("Dataset mixto total:", len(mixed))
print("Proporción generated:", len(selected_ds) / len(mixed))
print("Proporción human:", len(human_sample) / len(mixed))

mixed.to_json(MIXED_OUT_JSONL, orient="records", lines=True, force_ascii=False)
print("Dataset mixto guardado en:", MIXED_OUT_JSONL)

# Carga del modelo base cuantizado en 4 bits para reducir el uso de memoria GPU.
model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
ITER0_LORA_PATH = f"{MODELS_PATH}/iter4_academic"

model = PeftModel.from_pretrained(
    base_model,
    ITER0_LORA_PATH,
    is_trainable=True
)

model.print_trainable_parameters()

selected_ds = load_dataset("json", data_files=SELECTED_JSONL, split="train")
human_ds = load_dataset("json", data_files=HUMAN_TRAIN_JSON, split="train")

print("Selected:", len(selected_ds))
print("Human:", len(human_ds))


n_sel = len(selected_ds)
n_human = min(n_sel, len(human_ds))

human_sample = human_ds.shuffle(seed=42).select(range(n_human))

mixed = concatenate_datasets([selected_ds, human_sample]).shuffle(seed=42)

print("Dataset final tamaño:", len(mixed))
print("Proporción selected:", n_sel / len(mixed))

mixed.to_json(MIXED_OUT_JSONL, orient="records", lines=True, force_ascii=False)

split = mixed.train_test_split(test_size=0.2, seed=42)
train_ds = split["train"]
eval_ds = split["test"]

MAX_LEN = 512

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False
    )

train_tok = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
eval_tok  = eval_ds.map(tokenize_fn, remove_columns=eval_ds.column_names)





data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./results_iter5_academic",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,          # SOLO 1
    learning_rate=3e-5,          # LR MÁS BAJO
    fp16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    optim="paged_adamw_8bit",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator,
)


trainer.train()
metrics = trainer.evaluate()
print("Eval metrics:", metrics)

print("Train:", len(train_ds))
print("Eval:", len(eval_ds))

ITER_NUM = 5

SAVE_PATH = f"{MODELS_PATH}/iter{ITER_NUM}_academic"
os.makedirs(SAVE_PATH, exist_ok=True)

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Iter{ITER_NUM} guardado en: {SAVE_PATH}")

def generate_example(prompt="Abstract: In this study,", max_new_tokens=220):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prefix_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=120,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    gen = tokenizer.decode(out[0][prefix_len:], skip_special_tokens=True)
    gen = re.sub(r"\s+", " ", gen).strip()
    return gen

print(f"\n--- EJEMPLO GENERADO (Iter{ITER_NUM}) ---\n")
print(generate_example())
