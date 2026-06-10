

# Instalación de dependencias necesarias en Google Colab
!pip install -q transformers peft bitsandbytes pandas matplotlib tqdm torch scipy

import os
import re

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import ks_2samp

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from peft import PeftModel
from google.colab import drive
drive.mount("/content/drive")

print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

!pip install -q transformers peft pandas matplotlib tqdm torch

!pip install -U bitsandbytes

BASE_PATH = "/content/drive/MyDrive/TFG 2025-2026"
MODELS_PATH = f"{BASE_PATH}/models"
OUTPUT_PATH = f"{BASE_PATH}/generated_texts"
METRICS_PATH = f"{BASE_PATH}/metrics"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(METRICS_PATH, exist_ok=True)

ITERATION = N

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

LORA_PATH = f"{MODELS_PATH}/iter{ITERATION}_academic"
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print(f"Modelo Iter{ITERATION} cargado desde: {LORA_PATH}")

def is_complete(text: str) -> bool:
    text = text.strip()

    if not re.search(r"[\.!?]$", text):
        return False

    sentences = re.split(r"[\.!?]", text)
    if len([s for s in sentences if len(s.strip()) > 5]) < 3:
        return False

    if len(text.split()) < 120:
        return False

    return True

def generate_text(prefix: str, max_new_tokens: int = 220) -> str:
    model.eval()

    inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
      output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=140,
        do_sample=True,

        temperature=1.0,
        top_p=0.92,
        top_k=80,

        repetition_penalty=1.05,  #  bajar → más natural
        no_repeat_ngram_size=3,

        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if full.startswith(prefix):
        full = full[len(prefix):]

    full = re.sub(r"\s+", " ", full).strip()
    return full

def compute_perplexity(text: str) -> float:
    tokens = gltr_tok(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = tokens["input_ids"].to(gltr_model.device)

    with torch.no_grad():
        outputs = gltr_model(input_ids, labels=input_ids)
        loss = outputs.loss

    return float(torch.exp(loss).item())

def gpt_detector_score(text: str) -> float:
    inputs = det_tok(text, return_tensors="pt", truncation=True, max_length=512).to(det_model.device)
    with torch.no_grad():
        logits = det_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    return probs[0][1].item()

gltr_tok = AutoTokenizer.from_pretrained("gpt2")
gltr_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
gltr_model.eval()

det_model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
det_tok = AutoTokenizer.from_pretrained(det_model_name)
det_model = AutoModelForSequenceClassification.from_pretrained(det_model_name).to("cuda")
det_model.eval()

def gltr_scores(text: str):
    tokens = gltr_tok(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = tokens["input_ids"].to(gltr_model.device)

    with torch.no_grad():
        outputs = gltr_model(input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

    probs = torch.softmax(logits, dim=-1)
    ranks = []
    for i in range(labels.shape[1]):
        token_id = labels[0, i]
        rank = torch.sum(probs[0, i] > probs[0, i, token_id]).item()
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "gltr_top10_ratio": float(np.mean(ranks < 10)),
        "gltr_top100_ratio": float(np.mean(ranks < 100)),
        "gltr_mean_rank": float(np.mean(ranks))
    }

print(det_model.config.id2label)
print(det_model.config.label2id)

HUMAN_TRAIN_JSON = f"{BASE_PATH}/data_reference/train_split_iter0.json"
human_df = pd.read_json(HUMAN_TRAIN_JSON, lines=True)

print("Human dataset cargado:", len(human_df))
print(human_df.head())

tqdm.pandas(desc="Human Perplexity")
human_df["perplexity"] = human_df["text"].progress_apply(compute_perplexity)

tqdm.pandas(desc="Human GLTR")
human_df["gltr_mean_rank"] = human_df["text"].progress_apply(
    lambda t: gltr_scores(t)["gltr_mean_rank"]
)


human_ppl = human_df["perplexity"]
human_gltr = human_df["gltr_mean_rank"]

print(det_model.config.id2label)
print(det_model.config.label2id)


HUMAN_TRAIN_JSON = f"{BASE_PATH}/data_reference/train_split_iter0.json"
human_df.to_csv("human_metrics.csv", index=False)
mean_ppl = human_df["perplexity"].mean()
std_ppl  = human_df["perplexity"].std()

mean_gltr = human_df["gltr_mean_rank"].mean()
std_gltr  = human_df["gltr_mean_rank"].std()

print("HUMAN STATS:")
print("PPL:", mean_ppl, "±", std_ppl)
print("GLTR:", mean_gltr, "±", std_gltr)

abstract_prompts = [
    "Abstract:",
    "Abstract: In this study,",
    "Abstract: This paper investigates",
    "Abstract: We propose a novel framework for",
    "Abstract: The objective of this work is to",
    "Abstract: Recent advances in",
    "Abstract: We analyze the impact of",
    "Abstract: This research explores",
    "Abstract: A comprehensive analysis of",
    "Abstract: Experimental results demonstrate that"
    ]

N_GENERATIONS = 35
total = len(abstract_prompts) * N_GENERATIONS
print("Total textos a generar:", total)

HUMAN_METRICS_PATH = f"{BASE_PATH}/metrics/human_metrics.csv"
human_df.to_csv(HUMAN_METRICS_PATH, index=False)

print("TEST:", generate_text("Abstract:", max_new_tokens=50))

print(torch.cuda.is_available())
print(next(model.parameters()).device)

rows = []
id_counter = 0

for prompt in tqdm(abstract_prompts, desc="Generando abstracts"):
    for _ in range(N_GENERATIONS):
        gen = generate_text(prompt)

        rows.append({
            "iteration": ITERATION,
            "id": id_counter,
            "prompt": prompt,
            "generated_text": gen,
            "word_count": len(gen.split()),
        })
        id_counter += 1

df = pd.DataFrame(rows)

csv_full = f"{OUTPUT_PATH}/generated_iter{ITERATION}_abstracts.csv"
df.to_csv(csv_full, index=False)
print("CSV completo guardado en:", csv_full)

# FUNCIONES DE LIMPIEZA


def has_latex_artifacts(text: str) -> bool:
    if re.search(r"\\(cite|ref|label|begin|end)\{", text):
        return True
    if re.search(r"\$.*?\$", text):
        return True
    if re.search(r"\\\\[a-zA-Z]+", text):
        return True
    return False


def bigram_rep_ratio(text: str) -> float:
    toks = str(text).lower().split()
    if len(toks) < 10:
        return 1.0
    bigrams = list(zip(toks, toks[1:]))
    total = len(bigrams)
    uniq = len(set(bigrams))
    return float((total - uniq) / total) if total else 1.0


def ends_sentence(text: str) -> bool:
    text = str(text).strip()
    return (
        text.endswith((".", "!", "?")) or
        len(text.split()) > 120  # fallback
    )




# MÉTRICAS


df["bigram_rep_ratio"] = df["generated_text"].apply(bigram_rep_ratio)
df["latex_artifacts"] = df["generated_text"].apply(has_latex_artifacts)
df["ends_sentence"] = df["generated_text"].apply(ends_sentence)

tqdm.pandas(desc="GPT detector")
df["gpt_detector_score"] = df["generated_text"].progress_apply(gpt_detector_score)

tqdm.pandas(desc="GLTR")
gltr_results = df["generated_text"].progress_apply(gltr_scores)
df["gltr_mean_rank"] = gltr_results.apply(lambda x: x["gltr_mean_rank"])

tqdm.pandas(desc="Perplexity")
df["perplexity"] = df["generated_text"].progress_apply(compute_perplexity)





# FILTROS BÁSICOS


MIN_WORDS = 90
MAX_WORDS = 260
MAX_BIGRAM_REP = 0.30

filtered = df[
    (df["word_count"] >= MIN_WORDS) &
    (df["word_count"] <= MAX_WORDS) &
    (df["bigram_rep_ratio"] <= MAX_BIGRAM_REP) &
    (df["latex_artifacts"] == False) &
    (df["ends_sentence"] == True)
].copy()

# eliminar URLs y spam
filtered = filtered[
    ~filtered["generated_text"].str.contains(
        r"http|www\.|\.com|\.org|\.edu|github|doi|arxiv",
        case=False,
        regex=True
    )
]

filtered = filtered[
    ~filtered["generated_text"].str.contains(
        r"assignment|discount|order now|promo|coursework|essay",
        case=False,
        regex=True
    )
]

# detector IA
filtered = filtered[
    filtered["gpt_detector_score"] < 0.98
]

# Estrategia manual:
# los rangos de Perplexity y GLTR se fijan a partir del análisis exploratorio previo.
filtered = filtered[
    filtered["perplexity"].between(30, 60)
]

filtered = filtered[
    filtered["gltr_mean_rank"] >= 120
]

print("Tras filtros básicos:", len(filtered))



# SELECCIÓN POR CERCANÍA A HUMANO


# Z-score respecto a humanos
filtered["z_ppl"] = (filtered["perplexity"] - mean_ppl) / (std_ppl + 1e-8)
filtered["z_gltr"] = (filtered["gltr_mean_rank"] - mean_gltr) / (std_gltr + 1e-8)

# Distancia euclidiana
filtered["human_distance"] = np.sqrt(
    filtered["z_ppl"]**2 + (filtered["z_gltr"]**2)
)



# SELECCIÓN SUAVE (NUEVO)


DIST_THRESHOLD = 1.5

# Selección ponderada por cercanía al conjunto humano.
# Los textos con menor distancia tienen más probabilidad de ser seleccionados,
# pero se mantiene cierta diversidad en la muestra final.
filtered["selection_weight"] = 1 / (1 + filtered["human_distance"])
filtered["selection_weight"] = filtered["selection_weight"] / filtered["selection_weight"].sum()

# sampleo probabilístico
N_SAMPLES = 120  # ajusta según quieras

selected = filtered.sample(
    n=min(N_SAMPLES, len(filtered)),
    weights="selection_weight",
    random_state=42
).copy()

# ordenar por cercanía (solo para inspección)
selected = selected.sort_values("human_distance")

def cohens_d(sample, human):
    return (np.mean(sample) - np.mean(human)) / (np.std(human) + 1e-8)

d_ppl = cohens_d(selected["perplexity"], human_ppl)
d_gltr = cohens_d(selected["gltr_mean_rank"], human_gltr)

# KS test
ks_ppl = ks_2samp(selected["perplexity"], human_ppl)
ks_gltr = ks_2samp(selected["gltr_mean_rank"], human_gltr)

print("\n=== VALIDACIÓN ESTADÍSTICA ===")
print(f"Cohen PPL: {d_ppl:.3f}")
print(f"Cohen GLTR: {d_gltr:.3f}")

print(f"KS PPL: {ks_ppl.statistic:.3f} (p={ks_ppl.pvalue:.3f})")
print(f"KS GLTR: {ks_gltr.statistic:.3f} (p={ks_gltr.pvalue:.3f})")



# MÉTRICAS FINALES


print("Seleccionados finales:", len(selected))
print("Perplexity media:", selected["perplexity"].mean())
print("GLTR medio:", selected["gltr_mean_rank"].mean())

print("\n=== CERCANÍA HUMANA ===")
print("Distancia media:", filtered["human_distance"].mean())
print("% dentro del threshold:", (filtered["human_distance"] < DIST_THRESHOLD).mean())

print("\n=== CORRELACIÓN ===")
print(selected[["perplexity", "gltr_mean_rank"]].corr())

print("Fail word_count:",
      sum(~((df["word_count"] >= MIN_WORDS) &
            (df["word_count"] <= MAX_WORDS))))

print("Fail bigram_rep_ratio:",
      sum(df["bigram_rep_ratio"] > MAX_BIGRAM_REP))

print("Fail latex_artifacts:",
      sum(df["latex_artifacts"] == True))

print("Fail ends_sentence:",
      sum(df["ends_sentence"] == False))

MIN_SELECTED = 60

filtered = filtered.sort_values("human_distance", ascending=True)
top_n = int(len(filtered) * 0.25)
top_n = max(MIN_SELECTED, top_n)
top_n = min(top_n, len(filtered))

selected = filtered.head(top_n).copy()

csv_selected = f"{OUTPUT_PATH}/selected_iter{ITERATION}_top20_gltr.csv"
selected.to_csv(csv_selected, index=False)
print("Seleccionados guardados en:", csv_selected)


selected["source"] = "ai"
selected["label"] = 1

train_jsonl = f"{OUTPUT_PATH}/selected_iter{ITERATION}_for_iter{ITERATION+1}.jsonl"

selected[
    ["generated_text", "source", "label"]
].rename(
    columns={"generated_text": "text"}
).to_json(
    train_jsonl,
    orient="records",
    lines=True,
    force_ascii=False
)

print("JSONL para reentrenar guardado en:", train_jsonl)

filters = {
    "Longitud": len(df[(df["word_count"] >= MIN_WORDS) & (df["word_count"] <= MAX_WORDS)]),
    "Repetición": len(df[df["bigram_rep_ratio"] <= MAX_BIGRAM_REP]),
    "Sin LaTeX": len(df[df["latex_artifacts"] == False]),
    "Detector": len(df[df["gpt_detector_score"] < 0.98]),
    "Final": len(filtered)
}

plt.bar(filters.keys(), filters.values())
plt.xticks(rotation=45)
plt.title("Impacto de cada filtro")
plt.show()

plt.figure()
plt.hist(df["gltr_mean_rank"], bins=20)
plt.title(f"GLTR Mean Rank — Iter{ITERATION}")
plt.xlabel("gltr_mean_rank")
plt.ylabel("Frecuencia")
plt.show()

plt.figure()
plt.hist(selected["gltr_mean_rank"], bins=20)
plt.title(f"GLTR Mean Rank — Selected Top20% (Iter{ITERATION})")
plt.xlabel("gltr_mean_rank")
plt.ylabel("Frecuencia")
plt.show()

print("Notebook completado.")

print("Perplexity dataset completo:")
print(df["perplexity"].describe())

print("\nPerplexity seleccionados:")
print(selected["perplexity"].describe())

print("GLTR dataset completo:", df["gltr_mean_rank"].mean())
print("GLTR seleccionados:", selected["gltr_mean_rank"].mean())

print(selected[["perplexity","gltr_mean_rank"]].corr())
