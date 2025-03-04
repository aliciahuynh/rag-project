# from langchain.vectorstores import Pinecone
from pinecone import Pinecone
from transformers import BertTokenizer, BertModel
from langchain_openai import OpenAIEmbeddings

# pc = Pinecone(api_key="PINECONE_API_KEY")
pc = Pinecone(api_key="pcsk_JAgF7_A3pvgNDs3t1BBAey769QZCwuRLcDZ9dqFmgP4yHnZvWuRTRP7mnLxxy1aZ4qcFL")

index_name = "llama-2"
index = pc.Index(index_name) # connect to index
print("Connected to index")

# load Bert model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
print("loading model")

def embedding_function(query):
    encoded_input = tokenizer(query, return_tensors='pt')
    embeds = model(**encoded_input).pooler_output[0]
    return embeds

text_field = "text"  # the metadata field that contains our text

query = "What is so special about Llama 2?"
embed = embedding_function(query).tolist()
print(len(embed))

s = index.query(
    namespace="",
    vector=embed,
    top_k=1,
    include_values=False,
    include_metadata=True,
)
print(s)
text = s['matches'][0]['metadata']['text']
print(text)