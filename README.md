---
title: rag_project
app_file: main.py
sdk: gradio
sdk_version: 5.21.0
---

# rag_project

This project demonstrates a simple Retrieval-Augmented Generation (RAG) question answering system built with [Gradio](https://gradio.app/).

## System Architecture Overview

The implementation consists of a `SimpleQASystem` class that orchestrates three main components:

- **Semantic search** using [Sentence Transformers](https://www.sbert.net/) to compare the user's question against stored answers.
- **Answer generation** using the `t5-small` model.
- **Web search fallback** using [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) when the stored answers do not adequately match the question.

## Dataset Preparation

The system stores only example answers. Each answer is embedded with the `paraphrase-MiniLM-L6-v2` encoder.

```python
data = [
    {"answer": "The capital of France is Paris."},
    {"answer": "The largest planet in our solar system is Jupiter."},
    {"answer": "The chemical symbol for water is H2O."},
    {"answer": "EPAM established in 1993."},
    {"answer": "EPAM CEO is Arkadiy Dobkin"}
]
qa_system.prepare_dataset(data)
```

## Question Answering Flow

1. The question is encoded and compared against the stored answer embeddings using cosine similarity.
2. If the best similarity score is **0.7 or higher**, the corresponding stored answer becomes the context.
3. If the score is **below 0.7**, the system performs a DuckDuckGo web search and uses the retrieved snippets as context.
4. The selected context and question are passed to `t5-small` to generate the final answer.
5. Duplicate consecutive words are removed and the answer is capitalized.

The final response also includes debug information that reports the similarity score and the context source.

## Running the App

Launch the Gradio interface:

```bash
python main.py
```

This will start a small demo interface where you can type questions and receive answers.

## Limitations and Potential Improvements

- All embeddings are kept in memory; scaling to larger datasets would benefit from a vector database.
- Answer quality depends on the provided example answers and the quality of web search results.
- The current implementation runs on CPU only and may be slow for heavy workloads.

