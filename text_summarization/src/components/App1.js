import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import './App.css';
import axios from 'axios';
const FileUpload = () => {
  const [files, setFiles] = useState([]);
  const [extractedText, setExtractedText] = useState([]);
  const onDrop = (acceptedFiles) => {
    setFiles([...files, ...acceptedFiles]);
  };

  const removeFile = (index) => {
    const updatedFiles = [...files];
    updatedFiles.splice(index, 1);
    setFiles(updatedFiles);
  };

  // const handleUpload = () => {
  //   console.log("ehfg");
  //   // Add your upload logic here
  //   console.log('Uploading files:', files);
  //   // Reset files after uploading
  //   setFiles([]);
  // };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: '.pdf',  // Only accept PDF files
  multiple: true 
  });
  const handleSummarize = async () => {
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append('file', file.file));
      console.log(files);
  
      // Make a POST request to your Flask backend
      const response = await axios.post('http://localhost:5000/extract_text', formData);
  
      // Assuming the response is JSON and contains 'text_per_page'
      const { text_per_page } = response.data;
  
      // Set the extracted text to the state
      setExtractedText(text_per_page);
      
      // Display the text in the frontend (you may want to use a more suitable UI component)
      text_per_page.forEach((pageText, pageIndex) => {
        console.log(`Page ${pageIndex + 1} Text:`, pageText);
      });
    } catch (error) {
      console.error('Error during file upload:', error);
      // Handle error in the frontend (e.g., display an error message)
    }
  };
  return (
    <div>
      <div {...getRootProps()} className="dropzone">
        <input {...getInputProps()} />
        <p>Drag 'n' drop some files here, or click to select files</p>
      </div>
      <ul>
        {files.map((file, index) => (
          <li key={index}>
            {file.name}
            <span className="removeButton" onClick={() => removeFile(index)}>
              X
            </span>
          </li>
        ))}
      </ul>
      {files.length > 0 && (
        <button className="uploadButton" onClick={handleSummarize}>
          Upload
        </button>
      )}
      {extractedText.map((pageText, pageIndex) => (
              <div key={pageIndex}>
                <p>{`Page ${pageIndex + 1} Text:`}</p>
                <p>{pageText}</p>
              </div>
            ))}
    </div>
  );
};

function App1() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>File Upload Example</h1>
      </header>
      <main>
        <FileUpload />
      </main>
    </div>
  );
}

export default App;