# generate.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
from config import MODEL_NAME, OUTPUT_DIR, USE_GPU, LOAD_IN_4BIT

device = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"
print(f"Используется устройство: {device}")

base_model = MODEL_NAME
adapter_path = OUTPUT_DIR

# Конфигурация квантования (если использовалось при обучении)
bnb_config = None
if LOAD_IN_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    print("Используется 4-битное квантование для инференса")

# Проверка наличия адаптера
if not os.path.exists(adapter_path):
    print(f"Предупреждение: адаптер не найден в {adapter_path}. Будет использована базовая модель.")
    use_adapter = False
else:
    use_adapter = True

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
)

if use_adapter:
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Адаптер загружен.")
else:
    print("Используется базовая модель (без дообучения).")

model.eval()

prompt = "Продолжи рукопись 'Мертвые души. Том второй' в стиле Гоголя:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(  # type: ignore[union-attr]
        **inputs,
        max_new_tokens=300,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=" * 60)
print(result)
print("=" * 60)