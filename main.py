import gradio as gr
import os
# Set tokenizers parallelism before importing libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from duckduckgo_search import DDGS
import textwrap

from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def search_web(query, num_results=3):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=num_results)
        return [r['body'] for r in results]

def get_context_from_web(query):
    snippets = search_web(query)
    context = " ".join(snippets)
    return textwrap.fill(context, width=120)


class SimpleQASystem:
    def __init__(self):
        """Initialize QA system using T5"""
        try:
            # Use T5 for answer generation
            self.model_name = 't5-small'
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

            # Move model to CPU explicitly to avoid memory issues
            self.device = "cpu"
            self.model = self.model.to(self.device)  # type: ignore

            # Initialize storage
            self.answers = []
            self.answer_embeddings = None
            self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

            print("System initialized successfully")

        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    def prepare_dataset(self, data: List[Dict[str, str]]):
        """Prepare the dataset by storing answers and their embeddings"""
        try:
            self.answers = [item['answer'] for item in data]
            self.answer_embeddings = [self.encoder.encode(answer, convert_to_tensor=True) for answer in self.answers]
            for i, emb in enumerate(self.answer_embeddings):
                print(f"Embedding {i}: {emb.cpu().numpy().tolist()}")
            print("Dataset prepared successfully")
        except Exception as e:
            print(f"Dataset preparation error: {e}")
            raise

    def clean_answer(self, answer: str) -> str:
        """Clean the generated answer"""
        words = answer.split()
        cleaned_words = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() != words[i-1].lower():
                cleaned_words.append(word)
        cleaned = ' '.join(cleaned_words)
        return cleaned[0].upper() + cleaned[1:] if cleaned else cleaned

    def get_answer(self, question: str) -> str:
        """Get answer using semantic search and T5 generation"""
        try:
            if not self.answers or self.answer_embeddings is None:
                raise ValueError("Dataset not prepared. Call prepare_dataset first.")

            # Encode question using SentenceTransformer
            question_embedding = self.encoder.encode(
                question,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            # Move the question embedding to CPU (if not already)
            question_embedding = question_embedding.cpu()

            # Find most similar answer using cosine similarity
            similarities = cosine_similarity(
                question_embedding.numpy().reshape(1, -1),  # Use .numpy() for numpy compatibility
                np.array([embedding.cpu().numpy() for embedding in self.answer_embeddings])  # Move answer embeddings to CPU
            )[0]

            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            context = self.answers[best_idx]

            if best_score < 0.7:
                print("Low similarity score, performing web search via DuckDuckGo...")
                context = get_context_from_web(question)
                input_text = f"Given the context from web search (RAG similarity: {best_score:.4f}), what is the answer to the question: {question} Context: {context}"
            else:
                # Generate the input text for the T5 model
                input_text = f"Given the context (similarity: {best_score:.4f}), what is the answer to the question: {question} Context: {context}"

            print(input_text)
            # Tokenize input text
            input_ids = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).input_ids.to(self.device)

            # Generate answer with limited max_length
            outputs = self.model.generate(
                input_ids,
                max_length=50,  # Increase length to handle more detailed answers
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

            # Decode the generated answer
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Print the raw generated answer for debugging
            print(f"Generated answer before cleaning: {answer}")

            # Clean up the answer
            cleaned_answer = self.clean_answer(answer)
            return f"{cleaned_answer}\n\n[Debug Info] {input_text}"

        except Exception as e:
            print(f"Error in get_answer: {e}")
            return str(e)

# Create an instance of the QA system
qa_system = SimpleQASystem()

# Prepare the dataset (example data)
data = [
    {"answer": "The capital of France is Paris."},
    {"answer": "The largest planet in our solar system is Jupiter."},
    {"answer": "The chemical symbol for water is H2O."},
    {"answer": "EPAM established in 1993."},
    {"answer": "EPAM CEO is Arkadiy Dobkin"}

]
qa_system.prepare_dataset(data)

# Define the Gradio interface
def answer_question(question):
    return qa_system.get_answer(question)

iface = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="AI Multimodal Agent (RAG & WEB) Demo, Alex Uspenskiy 2025",
    description="Ask a question and get an answer based on the provided dataset.",
    allow_flagging="never"
)

# Launch the interface
iface.launch(share=True)
