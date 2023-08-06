from transformers import BertTokenizer, BertModel

def load_model_and_tokenizer () -> (BertModel, BertTokenizer):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return model, tokenizer