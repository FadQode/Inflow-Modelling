from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "arpanghoshal/EkmanClassifier"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

tokenizer.save_pretrained("./EkmanClassifier")
model.save_pretrained("./EkmanClassifier")
