const express = require('express');
const multer = require('multer');
const pdf = require('pdf-parse');
const cors = require('cors');
const app = express();
const port = 3001;
app.use(cors());
app.use(express.static('public'));

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.post('/upload', upload.array('pdfFiles'), async (req, res) => {
  console.log('In server express');
  const files = req.files;
  const textContents = [];
 
  for (const file of files) {
    try {
      const dataBuffer = file.buffer;
      const data = await pdf(dataBuffer);
      let textContent = data.text;

      // Remove unnecessary new line characters
      textContent = textContent.replace(/\n+/g, '\n');

      textContents.push({
        name: file.originalname,
        text: textContent,
      });
    } catch (error) {
      console.error(`Error processing ${file.originalname}: ${error.message}`);
      textContents.push({
        name: file.originalname,
        text: `Error processing the file: ${error.message}`,
      });
    }
  }

  const concatenatedText = textContents.map((file) => file.text).join('\n'); // Concatenate text

  console.log(concatenatedText);
  res.send(concatenatedText);
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
