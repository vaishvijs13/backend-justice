from transformers import pipeline

nli_model = pipeline("text-classification", model="roberta-large-mnli")

statement_1 = "I was at home all day."
statement_2 = "I was not at home all day."

result = nli_model({"text": statement_1, "text_pair": statement_2})
print(result)  # output should be [{'label': 'CONTRADICTION', 'score': 0.98}]
