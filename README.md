1️⃣ transformers

Hugging Face library for pretrained models and tokenizers.

Features used:

AutoTokenizer → Converts text into token IDs.

AutoModelForSequenceClassification → DistilBERT model for sequence classification.

TrainingArguments & Trainer → Simplifies training loops and evaluation.

2️⃣ peft

Stands for Parameter-Efficient Fine-Tuning.

Features used:

LoraConfig → Configures LoRA (Low-Rank Adaptation).

get_peft_model → Converts the model into a LoRA-enabled model.
4️⃣ torch

PyTorch, the deep learning framework.

Features used:

Model training and inference.

torch.no_grad() → Makes predictions without computing gradients.

torch.argmax() → Gets predicted class from logits.

5️⃣ numpy

For numerical computations.

Features used:

np.argmax() → Convert model logits to predicted labels.

Accuracy calculation.
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LoRA + DistilBERT Training & Prediction Flow</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.4.0/mermaid.min.js"></script>
    <script>mermaid.initialize({ startOnLoad: true });</script>
</head>
<body>
    <h2>LoRA + DistilBERT Data Flow: Training & Prediction</h2>
    <div class="mermaid">
        flowchart TD
            subgraph TRAINING
                A1[Text Input (IMDb Dataset)] --> B1[Tokenizer]
                B1 --> C1[LoRA Model]
                C1 --> D1[Trainer]
                D1 --> E1[Loss Calculation]
                E1 --> F1[Backpropagation & LoRA Weights Update]
                F1 --> D1
                D1 --> G1[Evaluation Metrics]
            end

            subgraph PREDICTION
                A2[New Text Input] --> B2[Tokenizer]
                B2 --> C2[LoRA Model (Trained)]
                C2 --> D2[Model.eval()]
                D2 --> E2[Output Logits]
                E2 --> F2[Prediction: POSITIVE / NEGATIVE]
            end

            %% Connect training to prediction
            G1 -.-> C2
    </div>
</body>
</html>
