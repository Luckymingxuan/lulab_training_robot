import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


def train_model():
    # 1. 加载数据（使用 JSON 文件）
    data_files = {
        "train": "train.jsonl",       # 训练集文件
        "validation": "valid.jsonl",  # 验证集文件
    }
    dataset = load_dataset("json", data_files=data_files)

    # 2. 加载模型和tokenizer
    model_name = "./qwen2_5-0_5b_model"  # 替换为你使用的模型名称，如 "Qwen/qwen-2.5-0.5b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 3. 将模型移到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 如果有GPU则打印出型号
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU now")

    # 4. 数据预处理
    def preprocess_function(examples):
        inputs = [p for p in examples['prompt']]
        outputs = [r for r in examples['response']]

        max_length = 80  # 统一设置输入和输出的最大长度

        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding='max_length')

        # 设置 labels，限制输出长度
        model_inputs["labels"] = tokenizer(outputs, max_length=max_length, truncation=True, padding='max_length')[
            "input_ids"]

        return model_inputs

    # 预处理数据集
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 5. 设置训练参数
    training_args = TrainingArguments(
        output_dir="./lulab-model-checkpoints",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        num_train_epochs=10,
        weight_decay=0.01,
    )

    # 6. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],  # 使用验证集进行评估
    )

    # 7. 开始训练
    trainer.train()

    # 8. 保存模型
    trainer.save_model("./lulab-model")
    tokenizer.save_pretrained("./lulab-model")  # 添加这行代码保存 tokenizer


# 调用函数开始训练
if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)  # 当前版本
    print("CUDA available:", torch.cuda.is_available())  # 是否有可用的CUDA支持
    print("CUDA version:", torch.version.cuda)  # PyTorch版本编译时所兼容的CUDA版本
    train_model()
