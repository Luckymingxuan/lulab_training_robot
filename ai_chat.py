from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./lulab-model"  # 本地模型路径


def generate_response(model_path, prompt, max_tokens=50, hint="You are a helpful assistant."):
    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 构建消息
    messages = [
        {"role": "system", "content": hint},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成响应
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 示例调用
hint = """# 角色
你是陆向谦直播间的专业客服，以简洁语言回复用户问题，字数不超 45 字。

## 技能
- 快速准确理解用户问题并作答。

## 限制
- 回复不超过 45 字。
"""

print("请输入对话内容，输入 'exit' 结束对话。")

while True:
    # 获取用户输入
    prompt = input("你: ")
    if prompt.lower() == 'exit':
        break

    # 生成对话回复，带有提示词
    response = generate_response(model_path, prompt, max_tokens=50, hint=hint)
    print("AI:", response)

