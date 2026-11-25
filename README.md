# Vector Search and Databases Assessment

## Project Overview

This repository contains the implementation of a four-task assessment covering vector search, vector databases, similarity search, database integration, and vector database management.

The project explores different approaches to storing, searching, retrieving, updating, and managing high-dimensional vector embeddings using Python and open-source technologies.

All tasks were implemented using locally available tools and services, avoiding paid cloud services.

## Technology Stack

- **Programming Language:** Python
- **Vector Search:** NumPy, FAISS
- **Relational Database:** PostgreSQL
- **Vector Extension:** pgvector
- **Vector Database:** Qdrant
- **Embedding Model:** all-minilm
- **Embedding Runtime:** Ollama
- **Data Processing:** NumPy
- **Environment:** Python virtual environment
- **Version Control:** Git and GitHub

---

# Task 1 – Vector Search with FAISS

## Objective

Implement and compare different vector search approaches using NumPy and FAISS.

## Implementation

- Generated 50,000 random vectors with 128 dimensions.
- Generated a query vector.
- Implemented exact nearest-neighbor search using NumPy.
- Implemented exact search using FAISS `IndexFlatL2`.
- Implemented approximate nearest-neighbor search using FAISS `IndexHNSWFlat`.
- Experimented with different HNSW parameters, including `M` and `efSearch`.
- Compared search latency and Recall@10.

## Results

| Search Method | Latency | Recall@10 |
|---|---:|---:|
| NumPy Exact Search | 14.5544 ms | 1.00 |
| FAISS Flat Search | 4.9988 ms | 1.00 |
| FAISS HNSW (M=32, efSearch=256) | 2.1781 ms | 1.00 |

The benchmark demonstrated the performance differences between brute-force exact search and graph-based approximate nearest-neighbor search for the tested query.

## Key Learnings

- Exact and approximate nearest-neighbor search.
- FAISS indexing methods.
- HNSW parameters and their impact on search performance.
- Recall@10 and search latency evaluation.

---

# Task 2 – Vector Search with PostgreSQL and pgvector

## Objective

Integrate vector embeddings into PostgreSQL using the pgvector extension and perform similarity searches using SQL.

## Implementation

- Installed and configured PostgreSQL with the pgvector extension.
- Created a PostgreSQL database named `vector_db`.
- Created a documents table containing document content, category, and 384-dimensional embeddings.
- Generated embeddings using Ollama and the all-minilm model.
- Inserted 20 documents into PostgreSQL.
- Implemented cosine similarity search using pgvector.
- Retrieved the top 3 most similar documents.
- Implemented category-based filtering.
- Created an HNSW index using `vector_cosine_ops`.
- Compared query execution plans using `EXPLAIN ANALYZE`.

## Results

- Successfully generated 20 document embeddings.
- Successfully inserted 20 documents.
- Retrieved the top 3 similar documents.
- Verified category-based filtering.
- Successfully created and verified the HNSW index.
- Examined execution plans before and after index creation.
- Used a diagnostic execution plan to verify HNSW index usage.

**Note:** With a small dataset of 20 rows, PostgreSQL may prefer a sequential scan even when an HNSW index exists. The index was separately verified using a diagnostic query plan.

## Key Learnings

- PostgreSQL vector storage.
- pgvector extension.
- Cosine similarity search.
- HNSW indexing.
- SQL filtering and query execution plans.

---

# Task 3 – Vector Search with Qdrant

## Objective

Create a vector database collection, store document embeddings, and perform semantic similarity searches using Qdrant.

## Implementation

- Generated embeddings using Ollama and the all-minilm model.
- Created a Qdrant collection named `assessment_collection`.
- Configured vector size to 384 dimensions.
- Configured cosine similarity.
- Prepared 30 documents across five categories:
  - HR
  - Finance
  - Technology
  - Support
  - Operations
- Stored documents with payload metadata:
  - content
  - source
  - page
  - category
- Inserted 30 points into the collection.
- Performed semantic similarity search using a natural-language query.
- Retrieved the top 5 matching documents.
- Verified persistent storage by reopening the collection.

## Results

- Collection created successfully.
- 30 document vectors inserted.
- Vector dimensions verified as 384.
- Cosine similarity search completed successfully.
- Top 5 search results retrieved.
- Persistent storage verified after reopening Qdrant.

## Key Learnings

- Qdrant collections and points.
- Vector dimensions and distance metrics.
- Payload metadata.
- Semantic similarity search.
- Local persistent vector database storage.

---

# Task 4 – Qdrant Vector Database Management

## Objective

Perform vector database management operations using Qdrant, including payload filtering, updating points, indexing, and deleting records.

## Implementation

### 1. Collection Management

- Connected to the original Task 3 collection.
- Verified that the original collection contained 30 points.
- Created a separate collection named `assessment_task4_collection`.
- Copied all 30 points into the Task 4 collection.

### 2. Payload Filtering

Applied the following filter:

- `category = HR`
- `page >= 3`

Successfully retrieved four matching records.

### 3. Upsert / Update

- Updated point ID 1.
- Replaced its content with revised HR leave-policy information.
- Generated a new embedding for the updated content.
- Verified the updated payload.
- Confirmed that the collection still contained 30 points.

### 4. Payload Indexing

Attempted to create a keyword payload index on the `category` field.

**Environment limitation:** Qdrant local mode does not provide functional payload indexing for this operation. The script reports this limitation instead of claiming successful index creation. A Qdrant server is required for functional payload indexing.

### 5. Delete by Filter

Applied the same HR and page filter to delete matching records.

- Records deleted: 4
- Remaining Task 4 points: 26

### 6. Data Integrity Verification

Verified that the original Task 3 collection remained unchanged with all 30 points.

## Results

| Operation | Result |
|---|---|
| Initial Task 4 points | 30 |
| Filtered records | 4 |
| Updated point | ID 1 |
| Deleted records | 4 |
| Final Task 4 points | 26 |
| Original Task 3 points | 30 |

## Key Learnings

- Qdrant collection management.
- Payload filtering.
- Point upsert and update.
- Delete operations using filters.
- Persistent storage and data integrity.
- Limitations of local Qdrant mode.

---

# Project Structure

```text
vector-search-databases-assessment/
│
├── task-1-faiss-search/
│   └── results/
│
├── task-2-pgvector/
│   ├── sql/
│   ├── screenshots/
│   └── postgres_demo.py
│
├── task-3-qdrant/
│   ├── screenshots/
│   └── qdrant_demo.py
│
├── task-4-qdrant-management/
│   ├── screenshots/
│   └── qdrant_management.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Local database storage, virtual environments, and generated cache files are excluded from Git.*

# Setup and Execution

## 1. Clone the Repository

```bash
git clone https://github.com/ajayharrieshtYT260949/vector-search-databases-assessment.git
cd vector-search-databases-assessment
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Ollama

Install Ollama and download the embedding model:

```bash
ollama pull all-minilm
```

Ensure Ollama is running before executing the embedding-related scripts.

## 5. Database Requirements

- Task 2 requires a running PostgreSQL instance with pgvector installed and configured.
- Tasks 3 and 4 use Qdrant local persistent storage.
- Tasks 3 and 4 do not require Docker or Qdrant Cloud.

Configure local database credentials and settings in a `.env` file where required. Never commit `.env` or database passwords to GitHub.

# Overall Learning Outcomes

Through these four tasks, the project demonstrates:

1. Exact and approximate vector similarity search.
2. Performance benchmarking using FAISS.
3. Vector storage and similarity search with PostgreSQL and pgvector.
4. Semantic search using embeddings and Qdrant.
5. Metadata-based filtering and retrieval.
6. HNSW indexing concepts and query execution analysis.
7. Vector database collection and point management.
8. Upsert, update, and delete operations.
9. Persistent storage and data integrity verification.
10. Practical implementation using open-source, locally running tools.

# Conclusion

This assessment provided practical experience with vector search algorithms, vector databases, embedding generation, similarity search, indexing, and database management.

The implementation uses Python, FAISS, PostgreSQL, pgvector, Qdrant, and Ollama to demonstrate different approaches to building and managing vector search systems.

## Repository

GitHub: https://github.com/ajayharrieshtYT260949/vector-search-databases-assessment