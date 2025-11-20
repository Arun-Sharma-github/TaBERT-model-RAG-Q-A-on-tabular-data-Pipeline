from table_bert import TableBertModel

model = TableBertModel.from_pretrained(
    "tabert_base_k1/model.bin",
)

print("Model initialized")
