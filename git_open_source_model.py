from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 模型名称
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 确保这是正确的模型名称

# 保存目录
save_directory = "./qwen2_5-0_5b_model"

# 创建保存目录（如果不存在）
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 下载并保存 tokenizer 和模型
try:
    print("下载模型和 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("保存 tokenizer...")
    tokenizer.save_pretrained(save_directory)
    print("保存模型...")
    model.save_pretrained(save_directory)

    print(f"模型和 tokenizer 已成功保存到 {save_directory}")

except Exception as e:
    print(f"发生错误: {e}")
