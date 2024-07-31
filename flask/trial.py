from io import BytesIO
import re
from PyPDF2 import PdfFileReader

def extract_text_from_pdfs(pdf_streams):
    all_text = ""
    try:
        for pdf_stream in pdf_streams:
            pdf_file = BytesIO(pdf_stream.read())
            pdf_reader = PdfFileReader(pdf_file)
            
            for page_num in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page_num].extract_text()
                # Remove newline characters
                page_text = page_text.replace('\n', ' ')
                # Remove bullet symbols (customize the pattern as needed)
                page_text = re.sub(r'\s*•\s*', ' ', page_text)
                
                all_text += page_text

    except Exception as e:
        all_text = f"Error extracting text: {str(e)}"
    
    return all_text

# Example usage
pdf_streams = [open('/Users/soundaryapoddaturi/Desktop/3-1/pp/1.CPU-GPU_1700337638211.pdf', 'rb')]  # Replace 'example.pdf' with your PDF file path
result_text = extract_text_from_pdfs(pdf_streams)
print(result_text)
