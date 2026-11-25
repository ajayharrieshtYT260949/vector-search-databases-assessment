
from pathlib import Path


import os
import numpy as np
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK4_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = "all-minilm"

SOURCE_COLLECTION = "assessment_collection"
TASK4_COLLECTION = "assessment_task4_collection"

SOURCE_STORAGE = PROJECT_ROOT / "task-3-qdrant" / "qdrant_storage"
TASK4_STORAGE = TASK4_DIR / "qdrant_storage"

VECTOR_SIZE = 384

FILTER = models.Filter(
    must=[
        models.FieldCondition(
            key="category",
            match=models.MatchValue(value="HR")
        ),
        models.FieldCondition(
            key="page",
            range=models.Range(gte=3)
        )
    ]
)


# --------------------------------------------------
# EMBEDDING GENERATION
# --------------------------------------------------

def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": OLLAMA_MODEL,
            "input": text
        },
        timeout=120
    )
    response.raise_for_status()

    data = response.json()
    vector = np.asarray(data["embeddings"][0], dtype=np.float32)

    if vector.shape != (VECTOR_SIZE,):
        raise ValueError(
            f"Expected {VECTOR_SIZE} dimensions, got {vector.shape}"
        )

    if not np.isfinite(vector).all():
        raise ValueError("Embedding contains invalid numeric values.")

    return vector.tolist()


# --------------------------------------------------
# COPY TASK 3 DATA INTO TASK 4 COLLECTION
# --------------------------------------------------

def copy_source_points() -> list[models.PointStruct]:
    print("\nConnecting to original Task 3 collection...")

    source_client = QdrantClient(path=str(SOURCE_STORAGE))

    try:
        if SOURCE_COLLECTION not in [
            item.name for item in source_client.get_collections().collections
        ]:
            raise RuntimeError("Original Task 3 collection was not found.")

        source_count = source_client.count(
            collection_name=SOURCE_COLLECTION,
            exact=True
        ).count

        if source_count != 30:
            raise RuntimeError(
                f"Expected 30 source points, found {source_count}"
            )

        print("Original collection verified:", source_count, "points")

        points, next_offset = source_client.scroll(
            collection_name=SOURCE_COLLECTION,
            limit=100,
            with_payload=True,
            with_vectors=True
        )

        if next_offset is not None:
            raise RuntimeError(
                "More than one page of points returned; "
                "extend the copy logic before continuing."
            )

        copied_points = []

        for point in points:
            if not isinstance(point.vector, list):
                raise RuntimeError(
                    f"Unexpected vector format for point {point.id}"
                )

            if len(point.vector) != VECTOR_SIZE:
                raise RuntimeError(
                    f"Incorrect vector size for point {point.id}"
                )

            copied_points.append(
                models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload or {}
                )
            )

        if len(copied_points) != 30:
            raise RuntimeError(
                f"Expected 30 copied points, got {len(copied_points)}"
            )

        return copied_points

    finally:
        source_client.close()


# --------------------------------------------------
# CREATE A CLEAN TASK 4 COLLECTION
# --------------------------------------------------

def create_task4_collection(
    points: list[models.PointStruct]
) -> QdrantClient:

    TASK4_STORAGE.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(TASK4_STORAGE))

    existing = [
        item.name for item in client.get_collections().collections
    ]

    # Reset only the Task 4 collection to make reruns repeatable.
    if TASK4_COLLECTION in existing:
        client.delete_collection(TASK4_COLLECTION)
        print("Previous Task 4 collection removed.")

    client.create_collection(
        collection_name=TASK4_COLLECTION,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )

    print("Task 4 collection created.")

    client.upsert(
        collection_name=TASK4_COLLECTION,
        points=points,
        wait=True
    )

    count = client.count(
        collection_name=TASK4_COLLECTION,
        exact=True
    ).count

    print("Copied points into Task 4:", count)

    if count != 30:
        raise RuntimeError("Task 4 copy verification failed.")

    return client


# --------------------------------------------------
# 1. PAYLOAD FILTERING
# --------------------------------------------------

def demonstrate_filtering(client: QdrantClient, query_vector: list[float]):

    print("\n========== 1. PAYLOAD FILTERING ==========")
    print("Condition: category = HR AND page >= 3")

    result = client.query_points(
        collection_name=TASK4_COLLECTION,
        query=query_vector,
        query_filter=FILTER,
        limit=10,
        with_payload=True,
        with_vectors=False
    ).points

    for point in result:
        print(
            f"ID: {point.id} | "
            f"Category: {point.payload['category']} | "
            f"Page: {point.payload['page']} | "
            f"Content: {point.payload['content']}"
        )

    print("Filtered result count:", len(result))

    if len(result) != 4:
        raise RuntimeError(
            f"Expected 4 filtered results, got {len(result)}"
        )

    return result


# --------------------------------------------------
# 2. UPSERT / UPDATE
# --------------------------------------------------

def demonstrate_upsert(client: QdrantClient):

    print("\n========== 2. UPSERT / UPDATE ==========")

    point_id = 1

    old_point = client.retrieve(
        collection_name=TASK4_COLLECTION,
        ids=[point_id],
        with_payload=True,
        with_vectors=True
    )

    if not old_point:
        raise RuntimeError("Point 1 was not found.")

    old_payload = old_point[0].payload or {}

    updated_content = (
        "Employees receive annual paid leave and can review "
        "the latest leave policy through the HR department."
    )

    updated_vector = generate_embedding(updated_content)

    updated_payload = dict(old_payload)
    updated_payload["content"] = updated_content
    updated_payload["updated_for_assessment"] = True

    client.upsert(
        collection_name=TASK4_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector=updated_vector,
                payload=updated_payload
            )
        ],
        wait=True
    )

    count = client.count(
        collection_name=TASK4_COLLECTION,
        exact=True
    ).count

    updated_point = client.retrieve(
        collection_name=TASK4_COLLECTION,
        ids=[point_id],
        with_payload=True,
        with_vectors=False
    )[0]

    print("Updated point ID:", point_id)
    print("Updated content:", updated_point.payload["content"])
    print("Updated marker:", updated_point.payload["updated_for_assessment"])
    print("Point count after upsert:", count)

    if count != 30:
        raise RuntimeError("Upsert unexpectedly changed the point count.")

    print("Upsert verification successful.")






# --------------------------------------------------
# 4. DELETE BY FILTER
# --------------------------------------------------

def demonstrate_delete(client: QdrantClient):

    print("\n========== 4. DELETE BY FILTER ==========")
    print("Deleting: category = HR AND page >= 3")

    client.delete(
        collection_name=TASK4_COLLECTION,
        points_selector=models.FilterSelector(filter=FILTER),
        wait=True
    )

    remaining = client.count(
        collection_name=TASK4_COLLECTION,
        exact=True
    ).count

    deleted = 30 - remaining

    print("Deleted points:", deleted)
    print("Remaining points:", remaining)

    if deleted != 4 or remaining != 26:
        raise RuntimeError("Delete verification failed.")

    print("Delete verification successful.")


# --------------------------------------------------
# 5. VERIFY TASK 3 IS UNCHANGED
# --------------------------------------------------

def verify_original_collection():

    print("\n========== 5. TASK 3 SAFETY CHECK ==========")

    client = QdrantClient(path=str(SOURCE_STORAGE))

    try:
        count = client.count(
            collection_name=SOURCE_COLLECTION,
            exact=True
        ).count

        print("Original Task 3 collection count:", count)

        if count != 30:
            raise RuntimeError(
                "Original Task 3 collection was unexpectedly changed."
            )

        print("Task 3 collection remains unchanged.")

    finally:
        client.close()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def demonstrate_payload_index(client):
    print("\n========== 3. PAYLOAD INDEXING ==========")

    try:
        client.create_payload_index(
            collection_name=TASK4_COLLECTION,
            field_name="category",
            field_schema="keyword",
        )

        collection_info = client.get_collection(TASK4_COLLECTION)
        payload_schema = collection_info.payload_schema or {}

        if "category" in payload_schema:
            print("Payload index created and verified.")
        else:
            print("Payload indexing is not functionally supported in local Qdrant mode.")
            print("This operation requires a Qdrant server.")

    except Exception as error:
        print("Payload indexing is unavailable in local mode.")
        print(f"Reason: {error}")

    print("Continuing with the remaining Task 4 operations...")

def main():

    print("=" * 60)
    print("TASK 4: QDRANT VECTOR DATABASE MANAGEMENT")
    print("=" * 60)

    print("\nGenerating search query embedding...")

    query_vector = generate_embedding(
        "What are the employee leave policies and health insurance benefits?"
    )

    copied_points = copy_source_points()

    client = create_task4_collection(copied_points)

    try:
        demonstrate_filtering(client, query_vector)
        demonstrate_upsert(client)
        demonstrate_payload_index(client)
        demonstrate_delete(client)

        final_count = client.count(
            collection_name=TASK4_COLLECTION,
            exact=True
        ).count

        print("\nFinal Task 4 collection count:", final_count)

        if final_count != 26:
            raise RuntimeError("Final collection count is incorrect.")

    finally:
        client.close()

    verify_original_collection()

    print("\n" + "=" * 60)
    print("TASK 4 COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
