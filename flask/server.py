# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import nltk
# import torch
# from sklearn.cluster import KMeans
# import numpy as np
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import re
# # import kaggle

# app = Flask(__name__)
# CORS(app)
# from langchain.llms import LlamaCpp
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# MODEL_PATH ="/Users/soundaryapoddaturi/Desktop/llama_proj/llamaenv/llama.cpp/models/7B/ggml-model-q4_0.bin"

# from sentence_transformers import SentenceTransformer,util
# def load_model()-> LLMChain:
#     try:
#         # kaggle.api.authenticate()
#         print("loading..........\n\n\n")
#         callback_manager = CallbackManager ( [StreamingStdOutCallbackHandler()])
#         Llama_model: LlamaCpp = LlamaCpp(
#         model_path=MODEL_PATH,
#         temperature=0.5,
#         max_tokens=2000,
#         verbose=True,
#         top_p=1,
#         callback_manager=callback_manager)
#         print("loaded\n\n\n")
#         # prompt: PromptTemplate=generate_prompt()
#         # llm_chain=LLMChain(llm=Llama_model,prompt=prompt)
#         return Llama_model
#     except Exception as e:
#         print(f"Error in loading model: {e}") 
#         return jsonify({'error': str(e)}), 500



# @app.route('/generateSummary', methods=['POST'])
# def generate_summary():
#     try:
#         print("SENDING TO .........")
#         llm = load_model()  # Load the model outside the loop
#         print("in server flask\n\n\n\n")
#         data = request.json
#         text = data.get('text', '')
#         initial_count = count_words(text)
#         print("initial count ", initial_count)
#         clustered_sentences = clustering(text)
#         total_cluster_summaries = []
#         for k, v in clustered_sentences.items():
#             print("cluster : ", k)
#             print(clustered_sentences[k])
#             text = " ".join(clustered_sentences[k])
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
#             print("AFTER CHUNKING\n\n\n")
#             chunks = text_splitter.split_text(text)
#             print(len(chunks))
#             cluster_chunk_summaries = []
#             for chunk in chunks:
#                 summary = generate_summary(chunk,llm)  # Use the loaded model
#                 torch.cuda.empty_cache()
#                 cluster_chunk_summaries.append(summary)
#             cluster_summary = "\n".join(cluster_chunk_summaries)
#             total_cluster_summaries.append(cluster_summary)

#         combined_cluster_summaries = "\n\n\n".join(total_cluster_summaries)
#         print(combined_cluster_summaries)
#         text = clean(combined_cluster_summaries)
#         print(text)
#         count_aftCnC = count_words(text)
#         print(count_aftCnC)

#         return jsonify({'processed_text': text})
#     except Exception as e:
#         print(f"Error in generateSummary: {e}")  # Enhanced error logging
#         return jsonify({'error': str(e)}), 500


# def clean(text):
#     cleaned_text = text.replace('\n\n', '\n')
#     cleaned_text = re.sub(r'[\d*•\-]+', ' ', cleaned_text)
#     return cleaned_text
# def count_words(text):
#     words = text.split()
#     num_words = len(words)
#     return num_words


# def clustering(text):
#     print("clustering.........\n\n\n\n\n")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     sentences_list = paragraph_to_sentences(text)
#     print("completed producing sentence_lists\n\n\n\n")
#     torch.cuda.empty_cache()
#     embedder = SentenceTransformer('all-MiniLM-L6-v2')
#     corpus_embeddings = embedder.encode(sentences_list)
#     corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
#     clustering_model = KMeans(n_clusters=4)
#     clustering_model.fit(corpus_embeddings)
#     cluster_assignment = clustering_model.labels_
#     print(cluster_assignment)
#     clustered_sentences = {}
#     for sentence_id, cluster_id in enumerate(cluster_assignment):
#         if cluster_id not in clustered_sentences:
#             clustered_sentences[cluster_id] = []
#         clustered_sentences[cluster_id].append(sentences_list[sentence_id])
#     return clustered_sentences
    
    
# import sys
# def paragraph_to_sentences(paragraph):
#     import nltk 
#     print(sys.path)
#     print("paragraph_to_sentences called \n\n\n\n")
#     nltk.download('punkt') 
#     print("bfhfg \n\n\n\n")
#     sentences = nltk.sent_tokenize(paragraph)
#     print("done paragraph_to_sentences")
#     return sentences


# # def process():
# #     try:
# #         print("TRYING TO LOAD THE MODEL\n\n\n\n")
# #         llm = CTransformers(model='/Users/soundaryapoddaturi/Desktop/llama_proj/llamaenv/llama.cpp/models/7B/ggml-model-q4_0.bin',
# #                         model_type='llama',
# #                         config={'max_new_tokens': 128,
# #                                 'temperature': 0.01}
# #                         )
# #         print("MODEL LOADED\n\n\n\n")
# #         return llm
        
#     # except Exception as e:
#     #     return jsonify({'error': str(e)})

# def generate_summary(text_chunk,Llama_model):
#     # Defining the template to generate summary
#     template = """
#     Generate a very brief and coherent summary of the given text, ensuring minimal word repetition and maintaining brevity
#     ```{text}```
#     SUMMARY:
#     """
#     prompt = PromptTemplate(template=template, input_variables=["text"])
#     llm_chain = LLMChain(prompt=prompt, llm=Llama_model)
#     summary = llm_chain.run(text_chunk)
#     torch.cuda.empty_cache()
#     return summary

# # Your other routes...

# if __name__ == "__main__":
#     app.run(debug=True, port=5001)


from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import torch
from sklearn.cluster import KMeans
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

app = Flask(__name__)
CORS(app, supports_credentials=True)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
MODEL_PATH ="/Users/soundaryapoddaturi/Desktop/llama_proj/llamaenv/llama.cpp/models/7B/ggml-model-q4_0.bin"

from sentence_transformers import SentenceTransformer,util
def load_model()-> LLMChain:
    try:

        print("loading..........\n\n\n")
        callback_manager = CallbackManager ( [StreamingStdOutCallbackHandler()])
        Llama_model: LlamaCpp = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        max_tokens=2000,
        verbose=True,
        top_p=1,
        callback_manager=callback_manager)
        print("loaded\n\n\n")
        # prompt: PromptTemplate=generate_prompt()
        # llm_chain=LLMChain(llm=Llama_model,prompt=prompt)
        return Llama_model
    except Exception as e:
        print(f"Error in loading model: {e}") 
        return jsonify({'error': str(e)}), 500



@app.route('/generateSummary', methods=['POST'])
def generate_summary():
    try:
        print("SENDING TO .........")
        llm = load_model()  # Load the model outside the loop
        print("in server flask\n\n\n\n")
        data = request.json
        text = data.get('text', '')
        initial_count = count_words(text)
        print("initial count ", initial_count)
        clustered_sentences = clustering(text)
        total_cluster_summaries = []
        for k, v in clustered_sentences.items():
            print("cluster : ", k)
            print(clustered_sentences[k])
            text = " ".join(clustered_sentences[k])
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
            print("AFTER CHUNKING\n\n\n")
            chunks = text_splitter.split_text(text)
            print(len(chunks))
            cluster_chunk_summaries = []
            for chunk in chunks:
                summary = generate_summary(chunk,llm)  # Use the loaded model
                torch.cuda.empty_cache()
                cluster_chunk_summaries.append(summary)
            cluster_summary = "\n".join(cluster_chunk_summaries)
            total_cluster_summaries.append(cluster_summary)

        combined_cluster_summaries = "\n\n\n".join(total_cluster_summaries)
        print(combined_cluster_summaries)
        text = clean(combined_cluster_summaries)
        print(text)
        count_aftCnC = count_words(text)
        print(count_aftCnC)

        return jsonify({'processed_text': text})
    except Exception as e:
        print(f"Error in generateSummary: {e}")  # Enhanced error logging
        return jsonify({'error': str(e)}), 500


def clean(text):
    cleaned_text = text.replace('\n\n', '\n')
    cleaned_text = re.sub(r'[\d*•\-]+', ' ', cleaned_text)
    return cleaned_text
def count_words(text):
    words = text.split()
    num_words = len(words)
    return num_words


def clustering(text):
    print("clustering.........\n\n\n\n\n")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences_list = paragraph_to_sentences(text)
    print("completed producing sentence_lists\n\n\n\n")
    torch.cuda.empty_cache()
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = embedder.encode(sentences_list)
    corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    clustering_model = KMeans(n_clusters=4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    print(cluster_assignment)
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append(sentences_list[sentence_id])
    return clustered_sentences
    
    
import sys
def paragraph_to_sentences(paragraph):
    import nltk 
    print(sys.path)
    print("paragraph_to_sentences called \n\n\n\n")
    nltk.download('punkt') 
    print("bfhfg \n\n\n\n")
    sentences = nltk.sent_tokenize(paragraph)
    print("done paragraph_to_sentences")
    return sentences


# def process():
#     try:
#         print("TRYING TO LOAD THE MODEL\n\n\n\n")
#         llm = CTransformers(model='/Users/soundaryapoddaturi/Desktop/llama_proj/llamaenv/llama.cpp/models/7B/ggml-model-q4_0.bin',
#                         model_type='llama',
#                         config={'max_new_tokens': 128,
#                                 'temperature': 0.01}
#                         )
#         print("MODEL LOADED\n\n\n\n")
#         return llm
        
    # except Exception as e:
    #     return jsonify({'error': str(e)})

def generate_summary(text_chunk,Llama_model):
    # Defining the template to generate summary
    template = """
    Generate a very brief and coherent summary of the given text, ensuring minimal word repetition and maintaining brevity
    ```{text}```
    SUMMARY:
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=Llama_model)
    summary = llm_chain.run(text_chunk)
    torch.cuda.empty_cache()
    return summary

# Your other routes...

if __name__ == "__main__":
    app.run(debug=True, port=5001)