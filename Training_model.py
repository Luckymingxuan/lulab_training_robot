from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import json
import torch


def main():
    # 模型保存目录
    save_directory = "./qwen2_5-0_5b_model"

    # 加载本地模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_directory, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(save_directory, trust_remote_code=True)

    # 将模型移到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 准备数据集（使用 JSON 文件）
    data_files = {
        "train": "train.jsonl",
        "validation": "valid.jsonl",
    }
    dataset = load_dataset("json", data_files=data_files)

    # 创建 LoRA 配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 应用 PEFT 微调模型
    model = get_peft_model(model, lora_config)

    # 定义 tokenizer 预处理函数
    def preprocess_function(examples):
        inputs = []
        for instruction, input_text in zip(examples["instruction"], examples["input"]):
            instruction = instruction.strip()
            input_text = input_text.strip()
            if input_text:
                prompt = f"指令：{instruction} 输入：{input_text} "
            else:
                prompt = f"指令：{instruction} "
            inputs.append(prompt)

        model_inputs = tokenizer(inputs, max_length=50, truncation=True, padding="max_length", return_tensors="pt")
        labels = tokenizer(examples["output"], max_length=50, truncation=True, padding="max_length", return_tensors="pt")["input_ids"]

        # 确保 labels 和输入对齐，且忽略 padding 的部分
        model_inputs["labels"] = labels.clone()
        model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
        return model_inputs

    # 处理数据集
    tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=4)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./lulab-model-checkpoints",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        push_to_hub=False,
        load_best_model_at_end=True
    )

    # 使用 Trainer API 进行微调
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 开始训练
    trainer.train()

    # 保存微调后的模型
    trainer.save_model("./lulab-model")

    # 生成 config.json
    config_file_path = "./lulab-model/config.json"
    with open(config_file_path, 'w') as f:
        json.dump(model.config.to_dict(), f)

    # 保存 tokenizer
    tokenizer.save_pretrained("./lulab-model")


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0))
    main()
