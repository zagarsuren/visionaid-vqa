
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-3-12b-it",
    repo_type="model",
    local_dir="./models/local_gemma3",
    local_dir_use_symlinks=False  # safer for moving folders around
)


"""from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# Hugging Face model ID
model_name = "google/gemma-3-12b-it"
# Local directory where the model will be saved
save_path = "./models/local_gemma3"

# Load processor and model from Hugging Face Hub
print(f"🔄 Downloading processor and model from: {model_name}")
processor = AutoProcessor.from_pretrained(model_name)
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",  # 👈 safely loads chunks onto CPU
    low_cpu_mem_usage=True,  # 👈 avoids huge RAM spike
    torch_dtype="auto"
)

# Save both to disk
print(f"💾 Saving to: {save_path}")
processor.save_pretrained(save_path)
model.save_pretrained(save_path)
print("✅ Gemma 3 model and processor saved locally!")"""
