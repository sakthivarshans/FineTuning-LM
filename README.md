# FineTuning-LM

A hands-on repository for learning and experimenting with **Large Language Model fine-tuning**, **LoRA-based Parameter-Efficient Fine-Tuning (PEFT)**, dataset formatting, tokenization, model training, and basic model inference using the Hugging Face ecosystem.

The repository also contains a separate machine-learning example for **house-price prediction using Linear Regression**, including data cleaning, visualization, and model evaluation.

---

## Overview

Fine-tuning adapts a pre-trained model to a specific task or domain by training it further on a smaller task-specific dataset.

This repository demonstrates two different levels of experimentation:

1. **Basic LLM fine-tuning workflow**
   - Load a pre-trained causal language model
   - Load a tokenizer
   - Prepare a dataset
   - Tokenize the dataset
   - Configure training arguments
   - Train the model
   - Save the fine-tuned model

2. **LoRA-based fine-tuning**
   - Load Qwen2.5-0.5B-Instruct
   - Prepare instruction-style question-answer data
   - Convert examples into a chat format
   - Tokenize the data
   - Apply LoRA adapters using PEFT
   - Train only a small number of parameters
   - Save the LoRA adapter
   - Run inference using the adapted model

The repository also includes a traditional machine-learning workflow using a house-price dataset.

---

## Repository Structure

```text
FineTuning-LM/
│
├── Fine_Tune_LLM_format.ipynb
│   └── Basic LLM fine-tuning template
│
├── LoRA_finetuning_code.ipynb
│   └── Complete LoRA fine-tuning workflow
│
├── linear_regression_house_prices.ipynb
│   └── Data cleaning, visualization and Linear Regression
│
├── house_prices_raw.csv
│   └── House-price dataset
│
└── README.md
    └── Project documentation
```

The current GitHub repository contains four tracked project files and does not currently include a dedicated README describing them.

---

# 1. Basic LLM Fine-Tuning

## File

```text
Fine_Tune_LLM_format.ipynb
```

This notebook demonstrates the general structure of a Hugging Face fine-tuning pipeline.

The notebook:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("llama-2-7b")
```

loads a causal language model and its tokenizer.

It then loads a dataset:

```python
from datasets import load_dataset

dataset = load_dataset("your_dataset")
```

and tokenizes the text:

```python
tokenized_data = tokenizer(
    dataset["text"],
    truncation=True,
    padding=True
)
```

Training configuration is created using:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
)
```

The Hugging Face `Trainer` is then used:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    eval_dataset=tokenized_data_val,
)

trainer.train()
```

Finally, the trained model is saved:

```python
trainer.save_model("fine-tuned-llm")
```

### Important Note

This notebook is currently a **template/example rather than a directly executable training pipeline**.

It contains placeholders:

```text
llama-2-7b
your_dataset
tokenized_data_val
```

Therefore, these values must be replaced with an actual compatible model, dataset, and validation dataset before running it successfully.

---

# 2. LoRA Fine-Tuning

## File

```text
LoRA_finetuning_code.ipynb
```

This is the main LLM fine-tuning implementation in the repository.

The notebook is configured to run with a GPU and was executed using a Tesla T4 environment.

The overall pipeline is:

```text
GPU
 ↓
Install dependencies
 ↓
Import libraries
 ↓
Load Qwen2.5-0.5B-Instruct
 ↓
Load tokenizer
 ↓
Load base model
 ↓
Create instruction dataset
 ↓
Format conversations
 ↓
Tokenize dataset
 ↓
Configure LoRA
 ↓
Apply LoRA
 ↓
Create data collator
 ↓
Configure training
 ↓
Train
 ↓
Save adapter
 ↓
Run inference
```

---

## 2.1 Dependencies

The notebook installs:

```bash
pip install -q -U transformers peft datasets accelerate
```

These libraries provide the main components required for the experiment.

### Transformers

Provides:

- Pre-trained models
- Tokenizers
- Training utilities
- Text generation

### PEFT

Provides Parameter-Efficient Fine-Tuning methods such as:

- LoRA
- Prefix Tuning
- Prompt Tuning
- Other adapter-based techniques

This repository specifically uses **LoRA**.

### Datasets

Provides the Hugging Face dataset abstraction used to construct and process the training dataset.

### Accelerate

Helps manage hardware acceleration and device placement.

### PyTorch

Provides the underlying deep-learning framework.

---

# 3. Base Model

The LoRA notebook uses:

```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
```

The model is loaded using:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

The model is therefore treated as a **Causal Language Model**, meaning it predicts the next token based on previous tokens.

The notebook also configures padding:

```python
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
```

The tokenizer uses the EOS token as the padding token when a padding token is not already defined:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

These steps are useful when preparing a causal language model for batched training.

---

# 4. Training Dataset

The notebook creates a small instruction dataset directly in Python.

The dataset contains questions and corresponding answers about Python concepts.

Example topics include:

```text
What is Python?
What is a Python list?
What is a Python dictionary?
What is a Python tuple?
What is a Python set?
What is a Python function?
What is a variable in Python?
What is a loop in Python?
What is recursion?
What is a class in Python?
```

The corresponding answers explain each concept in simple language.

The data is converted into a Hugging Face `Dataset`:

```python
dataset = Dataset.from_dict(data)
```

The structure is essentially:

```text
question                 answer
-----------------------------------------------
What is Python?          Python is...
What is a Python list?   A Python list is...
...
```

This is a very small demonstration dataset, not a production-scale training dataset.

---

# 5. Chat Formatting

The notebook converts each question-answer pair into a chat conversation.

The structure is:

```python
messages = [
    {
        "role": "user",
        "content": example["question"]
    },
    {
        "role": "assistant",
        "content": example["answer"]
    }
]
```

This is a **list containing dictionaries**.

The structure is:

```text
messages
   │
   └── List
       │
       ├── Dictionary
       │    ├── role: user
       │    └── content: question
       │
       └── Dictionary
            ├── role: assistant
            └── content: answer
```

The tokenizer's chat template is then applied:

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

This converts the structured conversation into the format expected by the Qwen model.

The resulting dataset contains a `text` field.

---

# 6. Tokenization

The formatted text is converted into tokens:

```python
def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )
```

The notebook then applies this function to the dataset:

```python
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)
```

Three important operations happen here.

### Truncation

```python
truncation=True
```

If an example is longer than the allowed sequence length, it is truncated.

### Maximum sequence length

```python
max_length=128
```

Each example is limited to 128 tokens.

### Padding

```python
padding="max_length"
```

Shorter examples are padded to the same length.

The result is a dataset suitable for model training.

---

# 7. LoRA Configuration

The notebook defines:

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

This is the core of the PEFT implementation.

## `r=8`

This is the **LoRA rank**.

LoRA does not directly update the original weight matrix.

Instead, it learns small low-rank matrices.

A simplified representation is:

```text
Original weight
      +
LoRA update
      ↓
Adapted layer
```

A lower rank means fewer trainable parameters.

---

## `lora_alpha=16`

This controls the scaling applied to the LoRA update.

The effective scaling is commonly related to:

```text
alpha / r
```

With the configuration:

```text
alpha = 16
r     = 8
```

the scaling factor is:

```text
16 / 8 = 2
```

---

## `target_modules`

The notebook targets:

```python
["q_proj", "v_proj"]
```

These correspond to the **Query** and **Value** projection layers in the Transformer attention mechanism.

Therefore, LoRA is not being added everywhere in the model.

It is specifically applied to these attention projections.

---

## `lora_dropout`

```python
lora_dropout=0.05
```

Applies dropout to the LoRA path during training.

This can help reduce overfitting.

---

## `bias`

```python
bias="none"
```

No bias parameters are trained through the LoRA configuration.

---

## `task_type`

```python
task_type="CAUSAL_LM"
```

Specifies that the model is being fine-tuned for causal language modeling.

---

# 8. Applying LoRA

The LoRA configuration is applied using:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Then:

```python
model.print_trainable_parameters()
```

is used to inspect how many parameters are actually being trained.

The notebook reports:

```text
trainable params: 540,672
all params: 494,573,440
trainable%: 0.1093
```

So only approximately **0.11% of the model parameters are trainable** in this experiment.

This is the key advantage of LoRA.

Instead of updating approximately 494 million parameters:

```text
Full fine-tuning
494M parameters → trainable
```

the experiment updates roughly:

```text
LoRA
0.54M parameters → trainable
```

while the original model weights remain frozen.

---

# 9. Data Collator

The notebook uses:

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
```

The important parameter is:

```python
mlm=False
```

This indicates that the training is for **causal language modeling**, not masked language modeling.

Conceptually:

```text
Previous tokens
       ↓
Predict next token
```

rather than:

```text
The cat [MASK] on the mat
          ↓
      Predict word
```

---

# 10. Training Configuration

The notebook uses:

```python
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    fp16=True
)
```

### Batch size

```python
per_device_train_batch_size=1
```

One training example is processed per device at a time.

### Gradient accumulation

```python
gradient_accumulation_steps=4
```

Gradients are accumulated for four steps before performing an optimizer update.

This effectively provides a larger batch without requiring all examples to be stored in GPU memory simultaneously.

### Epochs

```python
num_train_epochs=3
```

The dataset is processed three times.

### Learning rate

```python
learning_rate=2e-4
```

Controls how aggressively the LoRA parameters are updated.

### Logging

```python
logging_steps=1
```

Training information is logged every step.

### Saving

```python
save_strategy="no"
```

Automatic checkpoint saving is disabled during training.

### Mixed precision

```python
fp16=True
```

Uses 16-bit floating-point training, reducing GPU memory requirements and potentially increasing training speed on compatible NVIDIA GPUs.

These settings are present directly in the notebook.

---

# 11. Trainer

The Hugging Face `Trainer` handles the training loop:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
```

Training is started using:

```python
trainer.train()
```

The recorded experiment completed:

```text
Epochs: 3
Global steps: 9
Training loss: ~3.623
Training runtime: ~8.74 seconds
```

These are the results stored in the notebook execution output, not a general performance guarantee. The dataset is only ten examples, so the loss should not be interpreted as evidence of meaningful real-world model quality.

---

# 12. Saving the LoRA Model

The trained adapter is saved using:

```python
save_path = "./lora_finetuned_model"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
```

The notebook then verifies the saved files.

The resulting directory contains:

```text
lora_finetuned_model/
│
├── adapter_config.json
├── adapter_model.safetensors
├── chat_template.jinja
├── tokenizer_config.json
├── tokenizer.json
└── README.md
```

The important files are:

### `adapter_model.safetensors`

Contains the trained LoRA adapter weights.

### `adapter_config.json`

Contains the LoRA configuration.

### Tokenizer files

Used to reproduce the tokenization process required by the model.

This means the experiment saves the **adapter**, rather than creating a completely new full copy of the base model.

---

# 13. Inference

After training, the notebook defines:

```python
def ask_model(question):
```

This function accepts a question and generates a response using the trained model.

It first switches the model into evaluation mode:

```python
model.eval()
```

The question is converted into the same chat format used during training:

```python
messages = [
    {
        "role": "user",
        "content": question
    }
]
```

Then the chat template is applied:

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

The prompt is tokenized:

```python
inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)
```

Generation is performed with:

```python
output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)
```

Finally, the generated tokens are decoded back into text:

```python
generated_text = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)
```

The notebook tests the model with questions such as:

```python
ask_model("What is recursion?")
```

and:

```python
ask_model("What is a Python dictionary?")
```

The notebook contains recorded responses for these examples.

---

# 14. House Price Machine Learning Example

## File

```text
linear_regression_house_prices.ipynb
```

This notebook is separate from the LLM fine-tuning pipeline.

It demonstrates a traditional machine-learning workflow:

```text
CSV Dataset
    ↓
Load Data
    ↓
Inspect Data
    ↓
Clean Data
    ↓
Visualize Data
    ↓
Linear Regression
    ↓
Evaluate Model
    ↓
Plot Regression Line
```

The notebook imports:

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```

and loads:

```python
df = pd.read_csv("house_prices_raw.csv")
```

The dataset contains:

```text
House_ID
Area_sqft
Bedrooms
Age_years
Distance_to_City_km
City
Price
```

The CSV currently contains 154 data rows plus the header.

---

# 15. Data Inspection

The notebook checks:

```python
print(df.shape)
print(df.isna().sum())
print("Duplicates:", df.duplicated().sum())
```

This checks:

- Number of rows and columns
- Missing values
- Duplicate records

before cleaning.

---

# 16. Data Cleaning

The notebook performs:

```python
df = df.drop_duplicates()
```

Removes duplicate records.

Then:

```python
df["City"] = df["City"].str.strip().str.title()
```

Cleans city names by removing surrounding whitespace and standardizing capitalization.

Then:

```python
df = df.dropna()
```

Removes rows containing missing values.

Then:

```python
df = df[df["Age_years"] >= 0]
```

Removes records with negative house ages.

Then:

```python
df = df[df["Price"] < df["Price"].quantile(0.98)]
```

Removes approximately the highest 2% of prices.

Finally:

```python
df = df[df["Area_sqft"] < df["Area_sqft"].quantile(0.98)]
```

removes approximately the highest 2% of area values.

The notebook then reports the remaining dataset size and total number of missing cells.

---

# 17. Price Histogram

The notebook generates:

```python
plt.hist(df["Price"], bins=20)
```

This displays the distribution of house prices.

The chart helps understand:

- Price distribution
- Concentration of observations
- Possible extreme values

The notebook labels the chart:

```text
Price Histogram
```

with:

```text
X-axis → Price
Y-axis → Count
```


---

# 18. Average Price by City

The notebook calculates:

```python
avg = df.groupby("City")["Price"].mean()
```

This groups houses by city and calculates the average price for each city.

It then creates a bar chart:

```python
plt.bar(avg.index, avg.values)
```

This provides a basic comparison of average house prices across cities.

---

# 19. Linear Regression

The model uses:

```python
X = df[["Area_sqft"]]
y = df["Price"]
```

Therefore:

```text
Input  → Area_sqft
Output → Price
```

The model is created:

```python
model = LinearRegression()
```

and trained:

```python
model.fit(X, y)
```

The notebook then prints:

```python
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R2 score:", r2_score(y, model.predict(X)))
```

This gives:

### Slope

Represents the predicted change in price for a one-unit increase in area.

### Intercept

The theoretical predicted price when area is zero.

### R² score

Measures how much of the variation in the target is explained by the fitted linear relationship.

The notebook uses the same data for fitting and calculating R², so this is **training-set R², not a proper held-out test evaluation**. That distinction matters if this is being presented as an ML experiment rather than a classroom demonstration.

---

# 20. Regression Visualization

Finally, the notebook plots:

```python
plt.scatter(X, y, label="Data")
```

to show the original observations.

Then:

```python
plt.plot(
    X,
    model.predict(X),
    color="red",
    label="Regression line"
)
```

to show the fitted regression line.

The resulting visualization shows the relationship between:

```text
House Area ↔ House Price
```


---

# Technologies Used

## LLM / Deep Learning

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- Qwen2.5-0.5B-Instruct
- LoRA
- CUDA / NVIDIA GPU
- FP16 mixed precision

## Machine Learning

- Pandas
- Matplotlib
- Scikit-learn
- Linear Regression

---

# Installation

Clone the repository:

```bash
git clone https://github.com/sakthivarshans/FineTuning-LM.git
cd FineTuning-LM
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the main LLM dependencies:

```bash
pip install -U torch transformers peft datasets accelerate
```

For the house-price notebook:

```bash
pip install pandas matplotlib scikit-learn
```

For the LoRA notebook, a CUDA-enabled NVIDIA GPU is recommended. The existing experiment was run using a Tesla T4 GPU.

---

# Running the Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
Fine_Tune_LLM_format.ipynb
```

for the basic fine-tuning workflow.

For the actual LoRA experiment, open:

```text
LoRA_finetuning_code.ipynb
```

For the traditional ML example:

```text
linear_regression_house_prices.ipynb
```

Make sure:

```text
house_prices_raw.csv
```

is in the same directory as the Linear Regression notebook.

---

# Fine-Tuning Workflow

The core LLM workflow implemented in this repository can be summarized as:

```text
                 ┌─────────────────────┐
                 │ Pre-trained Qwen     │
                 │ Qwen2.5-0.5B-Instruct│
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Training Dataset    │
                 │ Question + Answer   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Chat Template       │
                 │ User → Assistant    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Tokenization        │
                 │ max_length = 128    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ LoRA Configuration  │
                 │ r=8, α=16           │
                 │ q_proj + v_proj     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ PEFT Model          │
                 │ ~0.11% trainable    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Trainer             │
                 │ 3 Epochs            │
                 │ LR = 2e-4           │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ LoRA Adapter        │
                 │ adapter_model...    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Inference           │
                 │ ask_model()         │
                 └─────────────────────┘
```

---

# LoRA vs Full Fine-Tuning

The repository demonstrates why LoRA is useful.

### Full Fine-Tuning

```text
Base Model
    ↓
Update millions/billions of parameters
    ↓
New model
```

This can require substantial memory and compute.

### LoRA Fine-Tuning

```text
Base Model
    ↓
Freeze original parameters
    +
Train small LoRA matrices
    ↓
LoRA Adapter
```

In this repository's experiment:

```text
Total parameters:       494,573,440
Trainable parameters:       540,672
Trainable percentage:        0.1093%
```

This is the central technical demonstration of the LoRA notebook.

---

# What This Repository Demonstrates

The repository currently covers:

- Pre-trained causal language models
- Hugging Face Transformers
- Tokenizers
- Chat templates
- Instruction datasets
- Dataset mapping
- Tokenization
- Causal language modeling
- Parameter-Efficient Fine-Tuning
- LoRA
- LoRA rank
- LoRA scaling
- Attention target modules
- LoRA dropout
- FP16 training
- Gradient accumulation
- Hugging Face Trainer
- Model saving
- LoRA adapter saving
- Text generation
- Basic inference
- Pandas data cleaning
- Data visualization
- Linear Regression
- R² evaluation

---

# Limitations

This repository is primarily an educational and experimental implementation.

The current LoRA experiment uses only **10 question-answer examples**, which is far too small to establish meaningful generalization or production-level fine-tuning quality.

The reported training loss:

```text
3.623
```

should therefore be treated as an experiment output rather than a model-quality benchmark.

The repository also does not currently provide:

- A large production dataset
- Train/validation/test splits for the LoRA experiment
- Automated evaluation metrics for the fine-tuned LLM
- Hyperparameter search
- Experiment tracking
- Checkpoint management
- Quantized QLoRA training
- Distributed training
- Production inference API
- Docker deployment
- Automated tests
- CI/CD
- A dedicated training script
- Configuration files for reproducible experiments

The basic fine-tuning notebook also contains placeholders and is not immediately executable without modification.

---

# Future Improvements

Possible extensions include:

1. Replace the toy dataset with a larger instruction dataset.

2. Add train/validation/test splits.

3. Add evaluation metrics appropriate to the task.

4. Add QLoRA using 4-bit quantization.

5. Experiment with different LoRA ranks:

```text
r = 4
r = 8
r = 16
r = 32
```

6. Experiment with additional target modules such as:

```text
q_proj
k_proj
v_proj
o_proj
```

7. Add hyperparameter configuration through YAML or JSON.

8. Convert the notebook into a reusable Python training script.

9. Add checkpointing and resume-from-checkpoint support.

10. Add experiment tracking using tools such as Weights & Biases.

11. Add proper evaluation before and after fine-tuning.

12. Add an inference API using FastAPI or Flask.

13. Add a simple web interface for interacting with the fine-tuned model.

14. Add model and dataset versioning.

---

# Learning Objectives

After working through this repository, you should understand the basic flow of LLM fine-tuning:

```text
Dataset
   ↓
Formatting
   ↓
Tokenizer
   ↓
Tokens
   ↓
Pre-trained LLM
   ↓
PEFT / LoRA
   ↓
Training
   ↓
Adapter
   ↓
Inference
```

You should also understand how traditional machine learning differs from LLM fine-tuning through the separate house-price regression example.

---

# Repository

GitHub:

https://github.com/sakthivarshans/FineTuning-LM

---

# Author

**Sakthivarshan S**

This repository is intended for learning, experimentation, and practical exploration of LLM fine-tuning and machine-learning workflows.
