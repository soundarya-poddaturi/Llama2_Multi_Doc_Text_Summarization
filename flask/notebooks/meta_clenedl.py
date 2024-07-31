!pip install -q transformers einops accelerate langchain bitsandbytes

!huggingface-cli login --token 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

!pip install sentencepiece

from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
from transformers import pipeline

model1 = ""
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

torch.cuda.empty_cache()

text

import re
def clean(text):
    cleaned_text = text.replace('\n\n', '\n')
    cleaned_text = re.sub(r'[\d*•\-]+', ' ', cleaned_text)
    return cleaned_text
def count_words(text):
    words = text.split()
    num_words = len(words)
    return num_words

initial_count=count_words(text)
print(initial_count)

!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer('all-MiniLM-L6-v2')

def paragraph_to_sentences(paragraph):
    # Using the nltk library for sentence tokenization
    import nltk
    nltk.download('punkt')  # Download the punkt tokenizer if not already downloaded

    # Tokenize the paragraph into sentences
    sentences = nltk.sent_tokenize(paragraph)

    return sentences

# Example usage
sentences_list = paragraph_to_sentences(text)

# Print the list of sentences
for idx, sentence in enumerate(sentences_list, start=1):
    print(f"Sentence {idx}: {sentence}")

torch.cuda.empty_cache()

from sklearn.cluster import KMeans
import numpy as np

embedder = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = embedder.encode(sentences_list)

# Normalize the embeddings to unit length
corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

# source: https://stackoverflow.com/questions/55619176/how-to-cluster-similar-sentences-using-bert

clustering_model = KMeans(n_clusters=4)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_
print(cluster_assignment)

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []
    clustered_sentences[cluster_id].append(sentences_list[sentence_id])

for k,v in clustered_sentences.items():
    print(k,end="\n\n\n")
    print(v,end="\n\n\n")
    x=" ".join (clustered_sentences[k])
    y=x.split()
    print(len(y))

# clustered_sentences[1]

def generate_summary(text_chunk):
    # Defining the template to generate summary
    template = """
    Generate a very brief and coherent summary of the given text, ensuring minimal word repetition and maintaining brevity
    ```{text}```
    SUMMARY:
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    summary = llm_chain.run(text_chunk)
    torch.cuda.empty_cache()
    return summary

import torch
from langchain import PromptTemplate, LLMChain
torch.cuda.empty_cache()
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

total_cluster_summaries = []  # Move this line outside the loop

for k, v in clustered_sentences.items():
    print("cluster : ", k)
    print(clustered_sentences[k])
    text = " ".join(clustered_sentences[k])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    print(len(chunks))
    i = 0
    cluster_chunk_summaries = []
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
    for chunk in chunks:
        summary = generate_summary(chunk)
        torch.cuda.empty_cache()
        cluster_chunk_summaries.append(summary)
        i = i + 1
        print(i)
    cluster_summary = "\n".join(cluster_chunk_summaries)
    total_cluster_summaries.append(cluster_summary)

combined_cluster_summaries = "\n\n\n".join(total_cluster_summaries)
print(combined_cluster_summaries)

text=clean(combined_cluster_summaries)
print(text)

count_aftCnC=count_words(text)
print(count_aftCnC)

from langchain.text_splitter import RecursiveCharacterTextSplitter




text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1084, chunk_overlap=150
)
chunks = text_splitter.split_text(text)
torch.cuda.empty_cache()
print(len(chunks))

chunk_summaries = []
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'# Set the max_split_size_mb environment variable
i=0
for chunk in chunks:
    summary = generate_summary(chunk)
    torch.cuda.empty_cache()
    chunk_summaries.append(summary)
    i=i+1
    print(i)
combined_summary = "\n".join(chunk_summaries)
text=clean(combined_summary)
print(count_words(text))
print(text)

def generate_summary(text_chunk):
    # Defining the template to generate summary
    template = """
    Summarize text very concisely in bullet points, avoiding word repetition for clarity
    ```{text}```
    SUMMARY:
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    summary = llm_chain.run(text_chunk)
    torch.cuda.empty_cache()
    return summary

text=clean(combined_cluster_summaries)
# print(text)
print(count_words(text))



from langchain.text_splitter import RecursiveCharacterTextSplitter




text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1084, chunk_overlap=150
)
chunks = text_splitter.split_text(text)
torch.cuda.empty_cache()
print(len(chunks))

chunk_summaries = []
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'# Set the max_split_size_mb environment variable
i=0
for chunk in chunks:
    summary = generate_summary(chunk)
    torch.cuda.empty_cache()
    chunk_summaries.append(summary)
    i=i+1
    print(i)
combined_summary = "\n".join(chunk_summaries)
text=clean(combined_summary)
print(count_words(text))
print(text)

from langchain.text_splitter import RecursiveCharacterTextSplitter



print(count_words(text))
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1084, chunk_overlap=150
)
chunks = text_splitter.split_text(text)
torch.cuda.empty_cache()
print(len(chunks))

chunk_summaries = []
torch.cuda.empty_cacadhe()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'# Set the max_split_size_mb environment variable
i=0
for chunk in chunks:
    summary = generate_summary(chunk)
    torch.cuda.empty_cache()
    chunk_summaries.append(summary)
    i=i+1
    print(i)
combined_summary = "\n".join(chunk_summaries)
text=clean(combined_summary)
print(count_words(text))
print(text)

print(combined_summary)
