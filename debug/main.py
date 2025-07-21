# train_grpo.py
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# Create a simple custom dataset for testing
def create_number_dataset(num_samples=256):
    prompts = []
    for i in range(num_samples):
        prompts.append(f"What number comes after {i}?")
    
    return Dataset.from_dict({"prompt": prompts})

dataset = create_number_dataset(num_samples=50)

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(5 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO")
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    per_device_train_batch_size=2,
    num_generations=2,
    generation_batch_size=2,
)
trainer.train()