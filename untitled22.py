import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# 1️⃣ Tiny demo dataset
data = {
    "text": [
        "I loved this movie, it was amazing",
        "This film was terrible and boring",
        "Excellent acting and great story",
        "Worst movie ever",
        "Very enjoyable and fun to watch",
        "I did not like the movie at all"
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

dataset = Dataset.from_dict(data)

# 2️⃣ Load tokenizer & model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3️⃣ Tokenize function
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64  # short reviews, fast training
    )

dataset = dataset.map(tokenize_function, batched=False)
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")

# 4️⃣ LoRA configuration
# Check correct target modules for DistilBERT
# DistilBERT attention linear layers: 'attention.q_lin', 'attention.v_lin'
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],  # DistilBERT's attention linear layers
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5️⃣ Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# 6️⃣ Train LoRA model
trainer.train()

# 7️⃣ Prediction function
def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)
    labels = ["NEGATIVE", "POSITIVE"]
    return labels[prediction.item()]

# 8️⃣ Test prediction
test_text = "The movie was fantastic and enjoyable"
print("Input:", test_text)
print("Prediction:", predict(test_text))
