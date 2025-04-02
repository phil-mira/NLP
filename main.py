import json
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load dataset
def load_preference_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }
    
    for item in data:
        dataset_dict["prompt"].append(item["question"])
        dataset_dict["chosen"].append(item["answers"][0])
        dataset_dict["rejected"].append(item["answers"][1])
    
    return dataset_dict

# Load and prepare the model with quantization
def prepare_model():
    # BitsAndBytes configuration for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer




def benchmark_model(model_path, tokenizer):
    # Load the fine-tuned model
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto"),
        model_path
    )
    
    # Load benchmark datasets
    benchmarks = {
        "ai2_arc": load_dataset("ai2_arc", "ARC-Challenge", split="test"),
        "gsm8k": load_dataset("gsm8k", "main", split="test"),
        "hellaswag": load_dataset("hellaswag", "default", split="validation"),
        "truthfulqa": load_dataset("truthful_qa", "multiple_choice", split="validation")
    }
    
    results = {}
    
    # Simple evaluation function
    def evaluate_on_dataset(dataset, task_type):
        correct = 0
        total = 0
        
        for sample in dataset:
            if task_type == "multiple_choice":
                # Handle multiple-choice questions like ARC
                question = sample["question"]
                choices = sample["choices"]["text"]
                
                best_score = -float('inf')
                best_idx = 0
                
                for i, choice in enumerate(choices):
                    prompt = f"Question: {question}\nAnswer: {choice}"
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        score = outputs.logits.mean().item()
                        
                    if score > best_score:
                        best_score = score
                        best_idx = i
                
                if best_idx == sample["choices"]["label"]:
                    correct += 1
                total += 1
                
            elif task_type == "math":
                # Handle math problems like GSM8K
                question = sample["question"]
                reference = sample["answer"]
                
                prompt = f"Solve this math problem step-by-step:\n{question}\n"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=200,
                        num_return_sequences=1,
                    )
                
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Naive answer extraction - in reality would need more sophisticated parsing
                if str(sample["answer"].split("####")[-1].strip()) in prediction:
                    correct += 1
                total += 1
            
            elif task_type == "hellaswag":
                # Handle HellaSwag dataset
                context = sample["ctx"]
                endings = sample["endings"]
                
                best_score = -float('inf')
                best_idx = 0
                
                for i, ending in enumerate(endings):
                    prompt = f"{context} {ending}"
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        score = outputs.logits.mean().item()
                        
                    if score > best_score:
                        best_score = score
                        best_idx = i
                
                if best_idx == sample["label"]:
                    correct += 1
                total += 1
            
            elif task_type == "truthfulqa":
                # Handle TruthfulQA dataset
                question = sample["question"]
                choices = sample["mc1_targets"]["choices"]
                
                best_score = -float('inf')
                best_idx = 0
                
                for i, choice in enumerate(choices):
                    prompt = f"Question: {question}\nAnswer: {choice}"
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        score = outputs.logits.mean().item()
                        
                    if score > best_score:
                        best_score = score
                        best_idx = i
                
                if best_idx == sample["mc1_targets"]["labels"].index(1):
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0
    
    # Evaluate on ARC
    results["ai2_arc"] = evaluate_on_dataset(benchmarks["ai2_arc"], "multiple_choice")
    
    # Evaluate on GSM8K
    results["gsm8k"] = evaluate_on_dataset(benchmarks["gsm8k"], "math")
    
    # Evaluate on HellaSwag
    results["hellaswag"] = evaluate_on_dataset(benchmarks["hellaswag"], "hellaswag")
    
    # Evaluate on TruthfulQA
    results["truthfulqa"] = evaluate_on_dataset(benchmarks["truthfulqa"], "truthfulqa")
    
    # Print results
    print("===== Benchmark Results =====")
    for benchmark, score in results.items():
        print(f"{benchmark}: {score:.4f}")
    
    # Save results to a JSON file
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    return results


def compare_models(trained_model_path, tokenizer, questions_file):
    # Load the trained model
    trained_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto"),
        trained_model_path
    )
    
    # Load the untrained GPT2 model
    untrained_model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
    
    # Load the questions
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    results = {"trained_model": {}, "untrained_model": {}}
    
    # Generate answers for each question
    for idx, question in enumerate(questions):
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate answer with trained model
        with torch.no_grad():
            trained_output = trained_model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
            )
        trained_answer = tokenizer.decode(trained_output[0], skip_special_tokens=True)
        results["trained_model"][f"question_{idx+1}"] = trained_answer
        
        # Generate answer with untrained model
        with torch.no_grad():
            untrained_output = untrained_model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
            )
        untrained_answer = tokenizer.decode(untrained_output[0], skip_special_tokens=True)
        results["untrained_model"][f"question_{idx+1}"] = untrained_answer
    
    # Save results to a JSON file
    with open("model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print("Comparison results saved to model_comparison_results.json")


def main():
    # Load preference dataset
    dataset_dict = load_preference_dataset("dataset.json")
    
    # Split into train/test
    train_size = int(0.9 * len(dataset_dict["prompt"]))
    
    train_dataset = {
        "prompt": dataset_dict["prompt"][:train_size],
        "chosen": dataset_dict["chosen"][:train_size],
        "rejected": dataset_dict["rejected"][:train_size]
    }
    
    eval_dataset = {
        "prompt": dataset_dict["prompt"][train_size:],
        "chosen": dataset_dict["chosen"][train_size:],
        "rejected": dataset_dict["rejected"][train_size:]
    }
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./dpo_gpt2_qlora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        report_to="tensorboard",
        save_strategy="steps",
        save_steps=500,
    )
    
    # Initialize DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Same as model but without LoRA
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=0.1,  # DPO beta parameter
        max_length=512,
        max_prompt_length=128,
    )
    
    # Train the model
    dpo_trainer.train()
    
    # Save the final model
    dpo_trainer.save_model("./final_dpo_gpt2_qlora")
    
    # Benchmark the model
    benchmark_model("./final_dpo_gpt2_qlora", tokenizer)

    compare_models("./final_dpo_gpt2_qlora", tokenizer, "./test_questions.json")

if __name__ == "__main__":
    main()