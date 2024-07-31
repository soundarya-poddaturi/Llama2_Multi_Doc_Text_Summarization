from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
import os
import kaggle
import subprocess

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Allow any origin and support credentials

@app.route('/generateSummary', methods=['POST'])
def generate_summary():
    try:
        print("SOUNDY")
        kaggle.api.authenticate()
        # with open('./meta_clenedl.ipynb', 'r') as notebook_file:
        #     notebook_code = notebook_file.read()
        # kernel_info = kaggle.api.kernels_push(
        #     "soundarya",
        #     notebook_code
        # )
        # kernel_id = kernel_info['id']
        # kaggle.api.kernels_output(kernel_id)
        notebook_path="/Users/soundaryapoddaturi/Desktop/llama2/flask/notebooks"
        upload_command = f'kaggle kernels push -p {notebook_path}'
        run_command = f'kaggle kernels run -p {notebook_path}'

        subprocess.run(upload_command, shell=True)
        subprocess.run(run_command, shell=True)

        # Retrieve the results (you may need to parse the notebook output)
        # Display the results on your web application
        return render_template('result.html', summary="summary")

    except Exception as e:
        print(f"Error in generateSummary: {e}")  # Enhanced error logging
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
