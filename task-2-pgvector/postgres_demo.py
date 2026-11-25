
import os
from pathlib import Path

import numpy as np
import psycopg
import requests

from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector


# --------------------------------------------------
# 1. LOAD CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "vector_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "all-minilm"
)

EMBEDDING_DIMENSIONS = 384
TOP_K = 3

SCHEMA_PATH = (
    Path(__file__).parent / "sql" / "schema.sql"
)


# --------------------------------------------------
# 2. SAMPLE DOCUMENTS
# --------------------------------------------------

DOCUMENTS = [
    {
        "content": "Employees receive annual paid leave according to company policy.",
        "category": "HR",
    },
    {
        "content": "The company provides health insurance benefits to employees.",
        "category": "HR",
    },
    {
        "content": "New employees must complete the onboarding process.",
        "category": "HR",
    },
    {
        "content": "Employee performance reviews are conducted every six months.",
        "category": "HR",
    },
    {
        "content": "The human resources department manages recruitment and hiring.",
        "category": "HR",
    },

    {
        "content": "Monthly financial reports summarize company revenue and expenses.",
        "category": "Finance",
    },
    {
        "content": "Employees must submit travel expenses for reimbursement.",
        "category": "Finance",
    },
    {
        "content": "The finance department prepares the annual budget.",
        "category": "Finance",
    },
    {
        "content": "Tax payments must follow government regulations.",
        "category": "Finance",
    },
    {
        "content": "Financial audits verify accounting records and transactions.",
        "category": "Finance",
    },

    {
        "content": "The engineering team maintains the application backend.",
        "category": "Technology",
    },
    {
        "content": "Database backups are performed regularly to prevent data loss.",
        "category": "Technology",
    },
    {
        "content": "Software developers use version control to manage source code.",
        "category": "Technology",
    },
    {
        "content": "Cloud infrastructure provides scalable computing resources.",
        "category": "Technology",
    },
    {
        "content": "Security updates protect systems against vulnerabilities.",
        "category": "Technology",
    },

    {
        "content": "Customers can contact support for assistance with orders.",
        "category": "Support",
    },
    {
        "content": "Support tickets are assigned to available service agents.",
        "category": "Support",
    },
    {
        "content": "Customer complaints should be resolved within the agreed timeframe.",
        "category": "Support",
    },
    {
        "content": "Technical support helps users troubleshoot application problems.",
        "category": "Support",
    },
    {
        "content": "Customers receive email notifications when support tickets are updated.",
        "category": "Support",
    },
]


# --------------------------------------------------
# 3. GENERATE OLLAMA EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(texts):

    endpoint = f"{OLLAMA_URL.rstrip('/')}/api/embed"

    print("\nGenerating embeddings with Ollama...")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Number of texts: {len(texts)}")

    response = requests.post(
        endpoint,
        json={
            "model": OLLAMA_MODEL,
            "input": texts,
        },
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()

    embeddings = np.asarray(
        data["embeddings"],
        dtype=np.float32,
    )

    expected_shape = (
        len(texts),
        EMBEDDING_DIMENSIONS,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}. "
            f"Expected: {expected_shape}"
        )

    print(f"Embedding shape: {embeddings.shape}")

    return embeddings


# --------------------------------------------------
# 4. DATABASE CONNECTION
# --------------------------------------------------

def connect_database():

    if not DB_CONFIG["password"]:
        raise ValueError(
            "DB_PASSWORD is missing. Check your root .env file."
        )

    print("\nConnecting to PostgreSQL...")

    conn = psycopg.connect(**DB_CONFIG)

    register_vector(conn)

    print("PostgreSQL connection successful!")

    return conn


# --------------------------------------------------
# 5. CREATE DATABASE SCHEMA
# --------------------------------------------------

def create_schema(conn):

    print("\nCreating database schema...")

    schema_sql = SCHEMA_PATH.read_text(
        encoding="utf-8"
    )

    statements = [
        statement.strip()
        for statement in schema_sql.split(";")
        if statement.strip()
    ]

    for statement in statements:
        conn.execute(statement)

    conn.commit()

    print("Schema created successfully!")


# --------------------------------------------------
# 6. INSERT DOCUMENTS
# --------------------------------------------------

def insert_documents(conn, embeddings):

    print("\nInserting documents...")

    conn.execute(
        "TRUNCATE TABLE documents RESTART IDENTITY"
    )

    insert_query = """
        INSERT INTO documents (
            content,
            category,
            embedding
        )
        VALUES (%s, %s, %s)
    """

    records = []

    for document, embedding in zip(
        DOCUMENTS,
        embeddings,
    ):

        records.append(
            (
                document["content"],
                document["category"],
                Vector(embedding.tolist()),
            )
        )

    with conn.cursor() as cur:
        cur.executemany(insert_query, records)

    conn.execute("ANALYZE documents")

    conn.commit()

    count = conn.execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]

    print(f"Documents inserted: {count}")


# --------------------------------------------------
# 7. EXACT COSINE SIMILARITY SEARCH
# --------------------------------------------------

def exact_search(conn, query_embedding):

    print("\n========== EXACT VECTOR SEARCH ==========")

    query = """
        SELECT
            id,
            content,
            category,
            embedding <=> %s AS cosine_distance
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT %s
    """

    vector = Vector(query_embedding.tolist())

    results = conn.execute(
        query,
        (vector, vector, TOP_K),
    ).fetchall()

    for row in results:

        print(f"\nID: {row[0]}")
        print(f"Content: {row[1]}")
        print(f"Category: {row[2]}")
        print(f"Cosine distance: {row[3]:.6f}")


# --------------------------------------------------
# 8. CATEGORY FILTERED SEARCH
# --------------------------------------------------

def category_search(conn, query_embedding):

    print("\n========== HR CATEGORY SEARCH ==========")

    query = """
        SELECT
            id,
            content,
            category,
            embedding <=> %s AS cosine_distance
        FROM documents
        WHERE category = %s
        ORDER BY embedding <=> %s
        LIMIT %s
    """

    vector = Vector(query_embedding.tolist())

    results = conn.execute(
        query,
        (
            vector,
            "HR",
            vector,
            TOP_K,
        ),
    ).fetchall()

    for row in results:

        print(f"\nID: {row[0]}")
        print(f"Content: {row[1]}")
        print(f"Category: {row[2]}")
        print(f"Cosine distance: {row[3]:.6f}")


# --------------------------------------------------
# 9. EXPLAIN ANALYZE
# --------------------------------------------------

def explain_query(conn, query_embedding, title):

    print(f"\n========== {title} ==========")

    query = """
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT id, content
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT %s
    """

    vector = Vector(query_embedding.tolist())

    results = conn.execute(
        query,
        (vector, TOP_K),
    ).fetchall()

    for row in results:
        print(row[0])


# --------------------------------------------------
# 10. CREATE HNSW INDEX
# --------------------------------------------------

def create_hnsw_index(conn):

    print("\n========== CREATING HNSW INDEX ==========")

    conn.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_documents_embedding_hnsw
        ON documents
        USING hnsw (embedding vector_cosine_ops)
    """)

    conn.execute("ANALYZE documents")

    conn.commit()

    print("HNSW index created successfully!")


# --------------------------------------------------
# 11. VERIFY INDEX
# --------------------------------------------------

def verify_index(conn):

    print("\n========== INDEX VERIFICATION ==========")

    query = """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'documents'
    """

    results = conn.execute(query).fetchall()

    for row in results:

        print(f"\nIndex: {row[0]}")
        print(f"Definition: {row[1]}")


# --------------------------------------------------
# 12. DEMONSTRATE HNSW QUERY PLAN
# --------------------------------------------------

def demonstrate_hnsw_plan(conn, query_embedding):

    print("\n========== HNSW INDEX PLAN ==========")

    # A 20-row table is small, so PostgreSQL may
    # reasonably prefer a sequential scan.
    #
    # Disable sequential scans locally for this
    # diagnostic demonstration.

    conn.execute("SET enable_seqscan = off")

    try:
        explain_query(
            conn,
            query_embedding,
            "HNSW INDEX EXECUTION PLAN",
        )

    finally:
        conn.execute("RESET enable_seqscan")

    conn.commit()


# --------------------------------------------------
# 13. MAIN EXECUTION
# --------------------------------------------------

def main():

    print("=" * 55)
    print("POSTGRESQL + PGVECTOR ASSESSMENT")
    print("=" * 55)

    texts = [
        document["content"]
        for document in DOCUMENTS
    ]

    embeddings = generate_embeddings(texts)

    query_text = (
        "What are the employee leave policies "
        "and health insurance benefits?"
    )

    query_embedding = generate_embeddings(
        [query_text]
    )[0]

    conn = None

    try:

        conn = connect_database()

        create_schema(conn)

        insert_documents(
            conn,
            embeddings,
        )

        exact_search(
            conn,
            query_embedding,
        )

        category_search(
            conn,
            query_embedding,
        )

        # Ensure an earlier run does not leave the
        # HNSW index in place before our comparison.

        conn.execute("""
            DROP INDEX IF EXISTS
            idx_documents_embedding_hnsw
        """)

        conn.execute("ANALYZE documents")
        conn.commit()

        explain_query(
            conn,
            query_embedding,
            "EXECUTION PLAN BEFORE HNSW",
        )

        create_hnsw_index(conn)

        verify_index(conn)

        explain_query(
            conn,
            query_embedding,
            "EXECUTION PLAN AFTER HNSW",
        )

        demonstrate_hnsw_plan(
            conn,
            query_embedding,
        )

        print("\n" + "=" * 55)
        print("TASK 2 COMPLETED SUCCESSFULLY")
        print("=" * 55)

    finally:

        if conn is not None:
            conn.close()
            print("\nPostgreSQL connection closed.")


if __name__ == "__main__":
    main()
 