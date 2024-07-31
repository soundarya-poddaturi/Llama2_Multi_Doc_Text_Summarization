import React, { useRef, useState } from 'react';
import axios from 'axios';
import './style.css';

const Main = () => {
  const fileInputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [filesUploaded, setFilesUploaded] = useState(false);
  const [allText, setAllText] = useState('');
  const [isSummarized, setIsSummarized] = useState(false);
  const [formData, setFormData] = useState(new FormData());

  const handleFileUpload = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    const selectedFiles = e.target.files;
    const newFiles = Array.from(selectedFiles);
  
    setFiles((prevFiles) => [...prevFiles, ...newFiles]);
  
    // Append new files to the existing FormData
    newFiles.forEach((file) => {
      formData.append('pdfFiles', file);
    });
  };
  

  const handleRemoveFile = (index) => {
    const updatedFiles = [...files];
    updatedFiles.splice(index, 1);
    setFiles(updatedFiles);

    if (updatedFiles.length === 0) {
      setFilesUploaded(false);
      setAllText('');
    } else {
      const allTextResult = updatedFiles.map((file) => file.text).join('\n');
      setAllText(allTextResult);
    }
  };

  const handleSummarize = async () => {
    try {
      if (files.length === 0 || isSummarized) {
        console.log('No files to summarize or already summarized.');
        return;
      }
      const response = await axios.post('http://localhost:3001/upload', formData);

      const concatenatedText = response.data;
      console.log(concatenatedText);
      try {
        console.log("soudy")
        const resback = await axios.post('http://localhost:5001/generateSummary', {
  text: concatenatedText,
}, {
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  }, 
});

        setAllText(resback.data);
        setFilesUploaded(true);
        setIsSummarized(true);
      } catch (error) {
        console.error('Error in generating summary:', error);
      }
    } catch (error) {
      console.error('Error uploading files:', error);
      // You can add a state to display user-friendly error messages
    }
  };

return (
  <div className='background-container'>
    <div className='translucent-layer'>
      <div className={`container ${filesUploaded ? 'files-uploaded' : ''}`}>
        <div className='total'>
          
          <div className='total_inner' onClick={handleFileUpload}>
          {files.map((file, index) => (
                <div key={index} className='file-entry'>
                  <p>{file.name}</p>
                  <button className="totalbtn" onClick={() => handleRemoveFile(index)}>
                    <span role="img" aria-label="Remove" className="remove-icon">x</span>
                  </button>
                </div>
              ))}
              <input
              type='file'
              accept='.pdf'
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileChange}
              multiple
            />

            <h1 className='browse'>Click Here</h1>
          </div>
        </div>
            
        <div>
          <div className='text-container'>
            <center><h2>{isSummarized ? 'Summarized Text:' : 'Add files to summarize........'}</h2></center>
            {isSummarized && <pre>{allText}</pre>}
          </div>
          <center>
            <button className="btn"onClick={handleSummarize}>Summarize</button>
          </center>
        </div>
      </div>
    </div>
  </div>
);
};

export default Main;
