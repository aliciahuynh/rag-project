from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm 
from transformers import BertTokenizer, BertModel

pc = Pinecone(api_key="PINECONE_API_KEY")

index_name = "llama-2"

print([ind['name'] for ind in pc.list_indexes()])
if index_name not in [ind['name'] for ind in pc.list_indexes()]:
    print("create index")
    pc.create_index(
        name=index_name,
        dimension=768, 
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
index = pc.Index(index_name) # connect to index

# load Bert model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

# load dataset
dataset = load_dataset(
    "jamescalam/llama-2-arxiv-papers-chunked",
    split="train"
)

batch_size = 25
data = dataset.to_pandas()

for i in tqdm(range(0, 200, batch_size)):
    # get batch
    i_end = min(200, i+batch_size)
    batch = data.iloc[i:i_end]
    # unique id
    ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
    # texts
    texts = [x['chunk'] for _, x in batch.iterrows()]
    # embeddings
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    embeds = model(**encoded_input).pooler_output
    vectors = [
        {'id': j,
         'metadata': {'text': c},
         'values': e} for j, c, e in zip(ids, texts, embeds)
    ]
    index.upsert(vectors=vectors)

# pc.delete_index(index_name)
