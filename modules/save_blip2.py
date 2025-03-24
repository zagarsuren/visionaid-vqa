from transformers import Blip2Processor, Blip2ForConditionalGeneration

model_name = "Salesforce/blip2-flan-t5-xl"
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name)

model.save_pretrained("./models/local_blip2")
processor.save_pretrained("./models/local_blip2")