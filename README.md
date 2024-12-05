# 目标
基于[Unsloth](https://docs.unsloth.ai/)框架微调Llama-3模型，部署，以及API调用

代码仓库见：[https://github.com/liaoyongzhi1010/Llama-tuning](https://github.com/liaoyongzhi1010/Llama-tuning)

# 微调模型
## 安装相关的库
```python
pip install unsloth
# Also get the latest nightly Unsloth!
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

## 加载模型
```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
```

> max_seq_length： 模型输入的最大长度
>
> dtype：选择适合当前硬件的精度
>
> load_in_4bit： 使用 4-bit 的量化
>
> 模型下载地址：
>
> ![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732633799986-0c16292a-ec15-48b5-8da6-7edc130c9cdb.png)
>

+ 模型列表：[https://huggingface.co/unsloth](https://huggingface.co/unsloth)

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732629638638-61fa066e-45d4-4a0a-96e4-6387501d317d.png)

## 测试原有模型
```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Input:
{}
### Response:
{}"""

FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "请用中文回答", # instruction
        "海绵宝宝的书法是不是叫做海绵体？", # input
        "", # output
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
```

> 模型的EOS_TOKEN由max_new_tokens控制的
>

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732630220074-73de5873-a35b-4c52-bfc6-cb96a08f11da.png)

## 准备微调数据集
```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("lyz1010/ruozhiba", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

> EOS_TOKEN：定义结束标记（End of Sequence），确保每个生成文本的末尾都有该标记，以避免模型进入无限生成循环。
>
> 数据集来源：[https://huggingface.co/datasets/lyz1010/ruozhiba](https://huggingface.co/datasets/lyz1010/ruozhiba)
>

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732631474661-d33ad79d-2e8c-472e-a599-339f3bda4441.png)

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732631434235-8d1ac221-3843-4e75-8794-d7585416d2ef.png)

## <font style="color:rgba(0, 0, 0, 0.87);">微调参数设置</font>
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
```

> r：控制低秩矩阵的秩（rank）
>
> target_modules：哪些模块（modules）会应用LoRA
>
> lora_alpha：LoRA中的一个缩放因子
>
> lora_dropout：dropout的概率
>
> bias：是否调整偏置项
>
> use_rslora：是否使用rank stabilized LoRA（RS-LoRA）的参数
>
> loftq_config：是否使用LoftQ配置
>

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732631798575-9118cd44-4acf-4500-8896-2d738f5ce4fb.png)

## 训练参数设置
```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732632410673-fec40caa-cf5e-4da5-972b-039e09e50c62.png)

## 训练模型
```python
trainer_stats = trainer.train()
```

## 保存模型
```python
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving
```

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732782876496-2704a14f-7c5a-4988-b5df-709bc82fad3f.png)

## 测试Lora模型
```python
from unsloth import FastLanguageModel

# 加载训练好的 LoRA 模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model",  # 使用你训练好的模型文件夹路径
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 启用 2x 更快的推理
FastLanguageModel.for_inference(model)

# 定义输入提示
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "只用中文回答问题",  # instruction
            "海绵宝宝的书法是不是叫做海绵体？",  # input
            "",  # output - 留空生成输出
        )
    ], return_tensors = "pt"
).to("cuda")

from transformers import TextStreamer

# 定义文本流输出
text_streamer = TextStreamer(tokenizer)

# 使用模型生成输出
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
```

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732778894171-baea54fb-50d9-40af-b1b9-3bbad580b41d.png)

# 部署模型
| **工具** | **特点** | **适用场景** |
| --- | --- | --- |
| **GGUF** | 轻量化、高效推理 | 低算力设备，资源受限场景 |
| **VLLM** | 高性能 GPU 推理 | 大规模任务，云端或企业级复杂应用 |
| **Ollama** | 对话优化，适合聊天 AI | 聊天机器人、智能客服系统 |
| **Troubleshooting** | 调试工具，问题定位与解决 | 开发和优化阶段，模型问题排查 |


选择具体工具时，应根据你的计算资源、任务需求和目标场景来决定。例如：

+ 如果你运行的是小型模型并目标是节省资源，可以选择 **GGUF**。
+ 如果需要处理高负载任务或使用 GPU，则选择 **VLLM**。
+ 如果目标是对话 AI 或交互式场景，选择 **Ollama**。
+ 在开发调试阶段，可以使用 **Troubleshooting** 提高效率。

## GGUF
```python
# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",
    )
```

> GGUF大模型UI工具：[https://www.nomic.ai/gpt4all](https://www.nomic.ai/gpt4all)
>

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732786663195-83b503e8-dcbb-4429-8d22-0bb9e276a900.png)

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732786676461-9fd5a0ca-5f0d-4e9c-8be1-f745f4ce3ed6.png)

## VLLM
```python
# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")
```

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732784235375-52b2c879-65d6-427d-a82f-a2996326e309.png)

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732784248530-af5cc5dd-cd84-4e44-8464-9c53ec9f1f19.png)

# 推理模型
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
```

![](https://cdn.nlark.com/yuque/0/2024/png/42936665/1732781646816-b135edd4-8754-41a9-a300-39fb5a6f0690.png)







