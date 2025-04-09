# LLaMA-SQL-3.1-Fine-Tuning-Meta-s-LLaMA-3.1-for-Natural-Language-to-SQL-Generation-using-QLoRA


A hands-on project where I fine-tuned Meta’s LLaMA 3.1 using **LoRA (Low-Rank Adaptation)** to enhance performance on **structured SQL reasoning tasks**. This demonstrates how LLMs can be adapted efficiently for domain-specific reasoning with minimal compute and maximum impact.

---

## 🚀 Project Highlights

✅ Fine-tuned a large language model (LLaMA 3.1) on custom SQL-based prompts  

✅ Used **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA** to reduce training cost  

✅ Created an Alpaca-style SQL dataset with instructions, inputs, outputs, and explanations  

✅ Evaluated performance before vs after fine-tuning using real-world SQL examples  

✅ Deployed models to GPU for inference and response comparison  

✅ Built end-to-end training and inference pipeline using Hugging Face + PyTorch

---

## 🧰 Tech Stack

| Component       | Tools / Libraries                       |
|----------------|------------------------------------------|
| Base Model      | Meta LLaMA 3.1 (open-weights)            |
| Fine-Tuning     | LoRA (via HuggingFace PEFT)              |
| Tokenization    | Alpaca-style multi-field instruction set |
| Frameworks      | `transformers`, `peft`, `trl`, `torch`   |
| Inference       | CUDA, `generate()`, tokenizer decode     |
| Platform        | Local GPU / Google Colab (CUDA enabled)  |

---


---
## 🧠 What is LoRA?

**LoRA (Low-Rank Adaptation)** enables efficient fine-tuning of large models by freezing most of the base parameters and only training small, low-rank matrices.

Benefits:

- ✅ Reduce training time and memory usage drastically  
- ✅ Avoid catastrophic forgetting by preserving base weights  
- ✅ Easy to plug-and-play on top of pretrained LLMs

---

## 📚 Dataset Format (Alpaca-Style)

Each sample contains:

- `instruction`: Table schema and content
- `input`: Natural language SQL query
- `output`: Ground truth SQL query
- `explanation`: Why the SQL query works

```json
{
  "instruction": "Company database: CREATE TABLE Patients(...);",
  "input": "SQL Prompt: How many patients have each diagnosis?",
  "output": "SELECT Diagnosis, COUNT(*) FROM Patients GROUP BY Diagnosis;",
  "explanation": "Group by diagnosis to count patients per category."
}

⚙️ How to Fine-Tune

Run the fine-tuning script with your dataset:

python finetune_lora.py \
  --base_model ./llama-3.1 \
  --dataset_path ./data/alpaca_sql_dataset.json \
  --output_dir ./checkpoints/lora-finetuned-llama \
  --batch_size 4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --num_epochs 3 \
  --cutoff_len 512

🔎 Inference & Evaluation

After training, compare base vs fine-tuned model performance:

python inference_compare.py
This script:

Loads both models

Formats input into SQL prompt structure

Runs .generate() on each model

Compares decoded outputs side by side

✅ Evaluation Results (Sample)

Prompt	Base LLaMA Output	Fine-Tuned Output

Patients table: count by diagnosis	Irrelevant or generic response	✅ SELECT Diagnosis, COUNT(*) ... GROUP BY ...

Orders table: total per customer	Missed aggregation logic	✅ Correct SQL with GROUP BY CustomerID

Transactions table: open-ended	Confused or verbose	✅ Concise, structured response
