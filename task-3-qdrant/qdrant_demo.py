
import os
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "all-minilm"
)

VECTOR_SIZE = 384
COLLECTION_NAME = "assessment_collection"

STORAGE_PATH = (
    Path(__file__).resolve().parent / "qdrant_storage"
)

TOP_K = 5


# ============================================================
# SAMPLE DATA: 30 DOCUMENTS
# ============================================================

DOCUMENTS = [
    # HR: 6 documents
    {
        "id": 1,
        "content": "Employees receive annual paid leave according to company policy.",
        "source": "employee_handbook.pdf",
        "page": 1,
        "category": "HR",
    },
    {
        "id": 2,
        "content": "The company provides health insurance benefits to employees.",
        "source": "employee_handbook.pdf",
        "page": 2,
        "category": "HR",
    },
    {
        "id": 3,
        "content": "New employees must complete the onboarding process.",
        "source": "employee_handbook.pdf",
        "page": 3,
        "category": "HR",
    },
    {
        "id": 4,
        "content": "Performance reviews are conducted every six months.",
        "source": "employee_handbook.pdf",
        "page": 4,
        "category": "HR",
    },
    {
        "id": 5,
        "content": "Employees can request flexible working arrangements.",
        "source": "employee_handbook.pdf",
        "page": 5,
        "category": "HR",
    },
    {
        "id": 6,
        "content": "The organization provides professional development training.",
        "source": "employee_handbook.pdf",
        "page": 6,
        "category": "HR",
    },

    # Finance: 6 documents
    {
        "id": 7,
        "content": "Employees must submit travel expenses for reimbursement.",
        "source": "finance_policy.pdf",
        "page": 1,
        "category": "Finance",
    },
    {
        "id": 8,
        "content": "Monthly financial reports must be submitted by the fifth day.",
        "source": "finance_policy.pdf",
        "page": 2,
        "category": "Finance",
    },
    {
        "id": 9,
        "content": "All business purchases require an approved purchase order.",
        "source": "finance_policy.pdf",
        "page": 3,
        "category": "Finance",
    },
    {
        "id": 10,
        "content": "The finance department monitors departmental budgets.",
        "source": "finance_policy.pdf",
        "page": 4,
        "category": "Finance",
    },
    {
        "id": 11,
        "content": "Vendor invoices must be verified before payment.",
        "source": "finance_policy.pdf",
        "page": 5,
        "category": "Finance",
    },
    {
        "id": 12,
        "content": "Annual financial statements are reviewed by auditors.",
        "source": "finance_policy.pdf",
        "page": 6,
        "category": "Finance",
    },

    # Technology: 6 documents
    {
        "id": 13,
        "content": "Developers must follow secure coding standards.",
        "source": "technology_guide.pdf",
        "page": 1,
        "category": "Technology",
    },
    {
        "id": 14,
        "content": "Database backups are performed every night.",
        "source": "technology_guide.pdf",
        "page": 2,
        "category": "Technology",
    },
    {
        "id": 15,
        "content": "Application programming interfaces use authentication tokens.",
        "source": "technology_guide.pdf",
        "page": 3,
        "category": "Technology",
    },
    {
        "id": 16,
        "content": "Software deployments must pass automated tests.",
        "source": "technology_guide.pdf",
        "page": 4,
        "category": "Technology",
    },
    {
        "id": 17,
        "content": "System administrators monitor server performance.",
        "source": "technology_guide.pdf",
        "page": 5,
        "category": "Technology",
    },
    {
        "id": 18,
        "content": "Access to production systems requires authorization.",
        "source": "technology_guide.pdf",
        "page": 6,
        "category": "Technology",
    },

    # Support: 6 documents
    {
        "id": 19,
        "content": "Customer support requests are tracked using ticket numbers.",
        "source": "support_manual.pdf",
        "page": 1,
        "category": "Support",
    },
    {
        "id": 20,
        "content": "Critical customer issues must be escalated immediately.",
        "source": "support_manual.pdf",
        "page": 2,
        "category": "Support",
    },
    {
        "id": 21,
        "content": "Support agents should respond to customers professionally.",
        "source": "support_manual.pdf",
        "page": 3,
        "category": "Support",
    },
    {
        "id": 22,
        "content": "Customer complaints are reviewed by the support manager.",
        "source": "support_manual.pdf",
        "page": 4,
        "category": "Support",
    },
    {
        "id": 23,
        "content": "Technical problems are assigned to specialized teams.",
        "source": "support_manual.pdf",
        "page": 5,
        "category": "Support",
    },
    {
        "id": 24,
        "content": "Customer feedback is used to improve service quality.",
        "source": "support_manual.pdf",
        "page": 6,
        "category": "Support",
    },

    # Operations: 6 documents
    {
        "id": 25,
        "content": "Daily operational activities must be recorded.",
        "source": "operations_manual.pdf",
        "page": 1,
        "category": "Operations",
    },
    {
        "id": 26,
        "content": "Equipment maintenance follows a scheduled maintenance plan.",
        "source": "operations_manual.pdf",
        "page": 2,
        "category": "Operations",
    },
    {
        "id": 27,
        "content": "Workplace safety inspections are performed regularly.",
        "source": "operations_manual.pdf",
        "page": 3,
        "category": "Operations",
    },
    {
        "id": 28,
        "content": "Inventory records must be updated after every stock movement.",
        "source": "operations_manual.pdf",
        "page": 4,
        "category": "Operations",
    },
    {
        "id": 29,
        "content": "Operational incidents must be reported to management.",
        "source": "operations_manual.pdf",
        "page": 5,
        "category": "Operations",
    },
    {
        "id": 30,
        "content": "Quality checks are required before products are dispatched.",
        "source": "operations_manual.pdf",
        "page": 6,
        "category": "Operations",
    },
]


# ============================================================
# VALIDATE SAMPLE DATA
# ============================================================

def validate_documents():
    if len(DOCUMENTS) != 30:
        raise ValueError(
            f"Expected 30 documents, got {len(DOCUMENTS)}"
        )

    ids = [document["id"] for document in DOCUMENTS]

    if len(ids) != len(set(ids)):
        raise ValueError("Document IDs must be unique.")

    required_fields = {
        "id",
        "content",
        "source",
        "page",
        "category",
    }

    for document in DOCUMENTS:
        if not required_fields.issubset(document):
            raise ValueError(
                f"Missing fields in document: {document}"
            )

        if not document["content"].strip():
            raise ValueError("Document content cannot be empty.")

    print("Document validation successful.")
    print(f"Total documents: {len(DOCUMENTS)}")


# ============================================================
# GENERATE EMBEDDINGS USING OLLAMA
# ============================================================

def generate_embeddings(texts):
    print("\nGenerating embeddings with Ollama...")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Number of texts: {len(texts)}")

    endpoint = f"{OLLAMA_URL.rstrip('/')}/api/embed"

    payload = {
        "model": OLLAMA_MODEL,
        "input": texts,
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=180,
        )

        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            "Could not generate embeddings. "
            "Check whether Ollama is running and the model is installed."
        ) from error

    data = response.json()

    if "embeddings" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response: {data}"
        )

    embeddings = np.asarray(
        data["embeddings"],
        dtype=np.float32,
    )

    expected_shape = (len(texts), VECTOR_SIZE)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Expected embedding shape {expected_shape}, "
            f"received {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    print(f"Embedding shape: {embeddings.shape}")

    return embeddings


# ============================================================
# CREATE OR LOAD QDRANT COLLECTION
# ============================================================

def create_collection(client):
    print("\nChecking Qdrant collection...")

    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection already exists: {COLLECTION_NAME}")

        collection_info = client.get_collection(COLLECTION_NAME)

        vector_config = collection_info.config.params.vectors

        if not isinstance(vector_config, VectorParams):
            raise ValueError(
                "Expected a single unnamed vector configuration."
            )

        if vector_config.size != VECTOR_SIZE:
            raise ValueError(
                f"Collection has {vector_config.size} dimensions; "
                f"expected {VECTOR_SIZE}."
            )

        if vector_config.distance != Distance.COSINE:
            raise ValueError(
                "Existing collection does not use cosine distance."
            )

        print("Existing collection configuration verified.")

    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print("Collection created successfully.")

    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Vector dimensions: {VECTOR_SIZE}")
    print("Distance metric: COSINE")


# ============================================================
# INSERT 30 POINTS INTO QDRANT
# ============================================================

def insert_points(client, embeddings):
    print("\nPreparing Qdrant points...")

    points = []

    for document, embedding in zip(DOCUMENTS, embeddings):
        point = PointStruct(
            id=document["id"],
            vector=embedding.tolist(),
            payload={
                "content": document["content"],
                "source": document["source"],
                "page": document["page"],
                "category": document["category"],
            },
        )

        points.append(point)

    print(f"Prepared points: {len(points)}")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    print("Points inserted successfully.")


# ============================================================
# VERIFY COLLECTION COUNT
# ============================================================

def verify_collection(client):
    print("\nVerifying Qdrant collection...")

    result = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )

    print(f"Stored point count: {result.count}")

    if result.count != 30:
        raise RuntimeError(
            f"Expected 30 points, but found {result.count}."
        )

    print("Collection count verification successful.")


# ============================================================
# SEMANTIC VECTOR SEARCH
# ============================================================

def search_documents(client, query_vector):
    print("\n========== QDRANT VECTOR SEARCH ==========")

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    )

    if not search_result.points:
        print("No search results found.")
        return

    for rank, point in enumerate(search_result.points, start=1):
        payload = point.payload or {}

        print(f"\nRank: {rank}")
        print(f"ID: {point.id}")
        print(f"Similarity score: {point.score:.6f}")
        print(f"Content: {payload.get('content')}")
        print(f"Source: {payload.get('source')}")
        print(f"Page: {payload.get('page')}")
        print(f"Category: {payload.get('category')}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TASK 3: QDRANT VECTOR DATABASE ASSESSMENT")
    print("=" * 60)

    validate_documents()

    STORAGE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_texts = [
        document["content"]
        for document in DOCUMENTS
    ]

    embeddings = generate_embeddings(document_texts)

    query_text = (
        "What are the employee leave policies "
        "and health insurance benefits?"
    )

    print(f"\nSearch query: {query_text}")

    query_embedding = generate_embeddings([query_text])[0]

    print("\nConnecting to local Qdrant storage...")

    client = QdrantClient(
        path=str(STORAGE_PATH)
    )

    try:
        print("Qdrant local connection successful!")

        create_collection(client)

        insert_points(
            client,
            embeddings,
        )

        verify_collection(client)

        search_documents(
            client,
            query_embedding,
        )

        print("\n" + "=" * 60)
        print("TASK 3 COMPLETED SUCCESSFULLY")
        print("=" * 60)

    finally:
        client.close()
        print("\nQdrant connection closed.")

    # Reopen storage to verify that the data persists.
    print("\nVerifying persistent storage...")

    verification_client = QdrantClient(
        path=str(STORAGE_PATH)
    )

    try:
        verify_collection(verification_client)

        print("Persistent storage verification successful!")

    finally:
        verification_client.close()


if __name__ == "__main__":
    main()
