import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from typing import Optional

def setup_model_and_trainer(
    model_name: str,
    train_dataset,
    eval_dataset=None,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    warmup_steps: int = 500,
    max_length: int = 512,
    output_dir: str = "./pythia-finetuned",
    gradient_accumulation_steps: int = 1,
    fp16: bool = True
):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if fp16 else torch.float32
    )

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        report_to="none"  # Disable all reporting
    )

    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    return trainer, model, tokenizer

def train_model(trainer):
    """Start the training process"""
    trainer.train()
    
def save_model(trainer, output_dir: str = "./final-model"):
    """Save the trained model"""
    trainer.save_model(output_dir)

if __name__ == "__main__":
    # Example usage (assuming you have your datasets ready):
    trainer, model, tokenizer = setup_model_and_trainer(
        model_name="EleutherAI/pythia-70m",
        train_dataset=train_dataset,  # Your training dataset
        eval_dataset=eval_dataset     # Your evaluation dataset
    )
    
    # Train the model
    train_model(trainer)
    
    # Save the final model
    save_model(trainer)
