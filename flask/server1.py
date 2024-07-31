from flask import Flask, render_template, request
# import kaggle.api
from dotenv import load_dotenv
import os
from flask_cors import CORS, cross_origin
app = Flask(__name__)
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
import kaggle
load_dotenv()
CORS(app, supports_credentials=True, origins='*')  # Allow any origin
@app.route('/summarize', methods=['POST'])
def summarize():
    print("vsdxgvxdg")
    input_text = request.form['input_text']
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    
    kaggle.api.authenticate(apikey=f'{kaggle_username}:{kaggle_key}')

    # Read notebook code from a file
    with open('./meta_clenedl.ipynb', 'r') as notebook_file:
        print("soundarya")
        notebook_code = notebook_file.read()

    # Create a Kaggle notebook
    kaggle.api.kernels_push_notebook_upload(
        "Your Kaggle Notebook Name",
        notebook_code,
        quiet=False
    )

    # Run the Kaggle notebook
    kaggle.api.kernels_output("Your Kaggle Notebook Name", quiet=False)

    # Retrieve the results (you may need to parse the notebook output)
    # Display the results on your web application
    return render_template('result.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)