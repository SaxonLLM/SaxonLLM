from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt

# Loading Dataset and Model
dataset = load_dataset("json", data_files="Scraped_Instruct(SI).jsonl")
dataset = dataset["train"].train_test_split(test_size=0.1)

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Apply LoRA
lora_config = LoraConfig(r=32, lora_alpha=64,lora_dropout=0.1, bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)

# Dataset
def preprocess_function(examples):
    inputs = [
        f"Instruction: {instr}\nResponse: {resp}" 
        for instr, resp in zip(examples["instruction"], examples["response"])
    ]
    model_inputs = tokenizer(inputs, max_length=1024, padding="max_length", truncation=True)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./SI",
    evaluation_strategy="epoch",   
    save_strategy="epoch",         
    logging_steps=100,             
    per_device_train_batch_size=3, 
    per_device_eval_batch_size=3,  
    gradient_accumulation_steps=12, 
    num_train_epochs=50,           
    learning_rate=3e-4,            
    lr_scheduler_type="cosine",    
    warmup_ratio=0.05,             
    fp16=True,                     
    optim="adamw_bnb_8bit",        
    logging_dir="./logs",
    report_to="none"
)

# Loss vs Epoch
class LossLogger(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.epochs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            self.epochs.append(state.epoch)

    def save_plot(self, filename="loss_vs_epoch_SI.png"):
        plt.plot(self.epochs, self.losses, marker='o', linestyle='-')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs. Epoch")
        plt.grid(True)
        plt.savefig(filename)


loss_logger = LossLogger()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    callbacks=[loss_logger] 
)

trainer.train()
loss_logger.save_plot("loss_vs_epoch_SI.png")

trainer.save_model("./SI")
tokenizer.save_pretrained("./SI")