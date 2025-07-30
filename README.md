---
title: rag_project
app_file: main.py
sdk: gradio
sdk_version: 5.21.0
---
System Architecture Overview

The implementation consists of a SimpleQASystem class that orchestrates two main components:
A semantic search system using Sentence Transformers
An answer generation system using T5

System DiagramCore Components

1. Initialization
   
def __init__(self):
    self.model_name = 't5-small'
    self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
    self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
    self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
The system initializes with two primary models:
T5-small: A smaller version of the T5 model for generating answers
paraphrase-MiniLM-L6-v2: A sentence transformer model for encoding text into meaningful vectors

2. Dataset Preparation
   
def prepare_dataset(self, data: List[Dict[str, str]]):
    self.answers = [item['answer'] for item in data]
    self.answer_embeddings = []
    for answer in self.answers:
        embedding = self.encoder.encode(answer, convert_to_tensor=True)
        self.answer_embeddings.append(embedding)
The dataset preparation phase:
Extracts answers from the input data
Creates embeddings for each answer using the sentence transformer
Stores both answers and their embeddings for quick retrieval

How the System Works

1. Question Processing
   
When a user submits a question, the system follows these steps:
Embedding Generation: The question is converted into a vector representation using the same sentence transformer model used for the answers.
Semantic Search: The system finds the most relevant stored answer by:
Computing cosine similarity between the question embedding and all answer embeddings
Selecting the answer with the highest similarity score

Context Formation: The selected answer becomes the context for T5 to generate a final response.

2. Answer Generation
   
def get_answer(self, question: str) -> str:
    # ... semantic search logic ...
    input_text = f"Given the context, what is the answer to the question: {question} Context: {context}"
    input_ids = self.tokenizer(input_text, max_length=512, truncation=True, 
                             padding='max_length', return_tensors='pt').input_ids
    outputs = self.model.generate(input_ids, max_length=50, num_beams=4, 
                                early_stopping=True, no_repeat_ngram_size=2
The answer generation process:
Combines the question and context into a prompt for T5
Tokenizes the input text with a maximum length of 512 tokens
Generates an answer using beam search with these parameters:

max_length=50: Limits answer length
num_beams=4: Uses beam search with 4 beams
early_stopping=True: Stops generation when all beams reach an end token
no_repeat_ngram_size=2: Prevents repetition of bigrams

3. Answer Cleaning

def clean_answer(self, answer: str) -> str:
    words = answer.split()
    cleaned_words = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() != words[i-1].lower():
            cleaned_words.append(word)
    cleaned = ' '.join(cleaned_words)
    return cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
Removes duplicate consecutive words (case-insensitive)
Capitalizes the first letter of the answer
Removes extra whitespace

Performance Considerations

Memory Management:

The system explicitly uses CPU to avoid memory issues
Embeddings are converted to CPU tensors when needed
Input length is limited to 512 tokens

Error Handling:

Comprehensive try-except blocks throughout the code
Meaningful error messages for debugging
Validation checks for uninitialized components

Usage Example

# Initialize system
qa_system = SimpleQASystem()
# Prepare sample data
data = [
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "What is the largest planet?", "answer": "The largest planet is Jupiter."}
]
# Prepare dataset
qa_system.prepare_dataset(data)
# Get answer
answer = qa_system.get_answer("What is the capital of France?")

Limitations and Potential Improvements

Scalability:

The current implementation keeps all embeddings in memory
Could be improved with vector databases for large-scale applications

Answer Quality:

Relies heavily on the quality of the provided answer dataset
Limited by the context window of T5-small
Could benefit from answer validation or confidence scoring

Performance:

Using CPU only might be slower for large-scale applications
Could be optimized with batch processing
Could implement caching for frequently asked questions

Conclusion
This implementation provides a solid foundation for a question-answering system, combining the strengths of semantic search and transformer-based text generation. 
While there's room for improvement, the current implementation offers a good balance between complexity and functionality, making it suitable for educational purposes and small to medium-scale applications.
