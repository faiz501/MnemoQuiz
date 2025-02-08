from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class PDFProcessor:
    """Handles text extraction from PDF files."""
    
    @staticmethod
    def extract_text(pdf_path):
        text = extract_text(pdf_path)
        return " ".join(text.split())  # Clean extracted text by removing extra spaces and newlines


class TextVectorizer:
    """Handles text vectorization using a sentence transformer model."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def vectorize(self, text):
        return self.model.encode(text)


class QuestionGenerator:
    """Handles question generation using a fine-tuned T5 model."""
    
    def __init__(self, model_name="valhalla/t5-base-qg-hl"):
        self.qg_pipeline = pipeline("text2text-generation", model=model_name)

    def generate_questions(self, text, max_questions=5):
        prompt = f"generate questions: {text}"  # Adding instruction for the model
        generated_output = self.qg_pipeline(prompt, max_length=128, num_return_sequences=max_questions)

        return [q["generated_text"] for q in generated_output]


def main():
    # File Path (Update Accordingly)
    pdf_path = "/Users/faiz/Desktop/PDF_ML/PDF_EXTRACT/Nptel_ConservationECO.pdf"

    # Step 1: Extract text from PDF
    extracted_text = PDFProcessor.extract_text(pdf_path)
    print("Extracted Text (Sample):\n", extracted_text[:500])  # Print a sample of extracted text

    # Step 2: Vectorize text
    vectorizer = TextVectorizer()
    vector = vectorizer.vectorize(extracted_text)
    print("\nVector Shape:", vector.shape)  # Print vector shape

    # Step 3: Generate questions
    qg = QuestionGenerator()
    questions = qg.generate_questions(extracted_text)

    print("\nGenerated Questions:")
    for idx, q in enumerate(questions, 1):
        print(f"Q{idx}: {q}")


if __name__ == "__main__":
    main()
