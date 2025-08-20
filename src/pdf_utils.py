# src/pdf_utils.py
def extract_text_from_pdf(uploaded_file):
    """
    uploaded_file can be a BytesIO (Streamlit) or path.
    Returns extracted text (string).
    """
    try:
        import fitz  # PyMuPDF
        # if streamlit uploaded file passed, it has .read(); create in-memory
        if hasattr(uploaded_file, "read"):
            data = uploaded_file.read()
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            doc = fitz.open(uploaded_file)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception:
        # fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            if hasattr(uploaded_file, "read"):
                # PyPDF2 can read BytesIO
                uploaded_file.seek(0)
                reader = PdfReader(uploaded_file)
            else:
                reader = PdfReader(uploaded_file)
            text = ""
            for p in reader.pages:
                text += p.extract_text() or ""
            return text
        except Exception as e:
            return ""
