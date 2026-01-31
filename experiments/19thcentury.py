from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # type of task to train on
    inference_mode=False, # set to False for training
    r=8, # dimension of the smaller matrices
    lora_alpha=32, # scaling factor
    lora_dropout=0.1 # dropout of LoRA layers
)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

model.add_adapter(lora_config,adapter_name = "lora_1")
trainer = Trainer(model=model,
train_dataset = 

)
trainer.train()