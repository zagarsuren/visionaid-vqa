from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_name = "google/paligemma-3b-ft-vqav2-448"

# Login first using: `huggingface-cli login`
processor = AutoProcessor.from_pretrained(model_name)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_name)

# Save locally
model.save_pretrained("./models/local_paligemma")
processor.save_pretrained("./models/local_paligemma")
