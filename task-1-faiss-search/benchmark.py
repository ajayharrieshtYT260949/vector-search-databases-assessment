
"""
Task 1: Exact vs Approximate Vector Search

Dataset:
    50,000 random vectors
    128 dimensions
    Top 10 nearest neighbours

Algorithms:
    1. NumPy exact search
    2. FAISS IndexFlatL2 (exact)
    3. FAISS IndexHNSWFlat (approximate)

Outputs:
    results/benchmark_results.csv
    results/recall_vs_latency.png
    results/search_comparison.png
"""

import csv
import time
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

NUM_VECTORS = 50_000
DIMENSIONS = 128
TOP_K = 10

RANDOM_SEED = 42
REPEATS = 10

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def measure_latency(search_function, repeats=REPEATS):
    """Measure search latency using median execution time."""

    # Warm-up
    search_function()

    timings = []

    for _ in range(repeats):
        start = time.perf_counter()

        search_function()

        elapsed = time.perf_counter() - start
        timings.append(elapsed * 1000)

    return float(np.median(timings))


def recall_at_k(exact_ids, approximate_ids):
    """Calculate Recall@K."""

    exact_set = set(exact_ids)
    approximate_set = set(approximate_ids)

    return len(exact_set & approximate_set) / len(exact_set)


# --------------------------------------------------
# 3. GENERATE RANDOM DATASET
# --------------------------------------------------

print("\n========== DATASET GENERATION ==========")

rng = np.random.default_rng(RANDOM_SEED)

vectors = rng.standard_normal(
    (NUM_VECTORS, DIMENSIONS)
).astype(np.float32)

query = rng.standard_normal(
    (1, DIMENSIONS)
).astype(np.float32)

print("Number of vectors:", vectors.shape[0])
print("Dimensions:", vectors.shape[1])
print("Query shape:", query.shape)
print("Top K:", TOP_K)


# --------------------------------------------------
# 4. NUMPY EXACT SEARCH
# --------------------------------------------------

print("\n========== NUMPY EXACT SEARCH ==========")


def numpy_exact_search():

    differences = vectors - query

    distances = np.einsum(
        "ij,ij->i",
        differences,
        differences
    )

    nearest_ids = np.argpartition(
        distances,
        TOP_K - 1
    )[:TOP_K]

    nearest_ids = nearest_ids[
        np.argsort(distances[nearest_ids])
    ]

    return nearest_ids


numpy_ids = numpy_exact_search()

numpy_latency = measure_latency(
    numpy_exact_search
)

print("Nearest IDs:", numpy_ids)
print(f"NumPy latency: {numpy_latency:.4f} ms")


# --------------------------------------------------
# 5. FAISS EXACT SEARCH - INDEXFLATL2
# --------------------------------------------------

print("\n========== FAISS EXACT SEARCH ==========")

flat_index = faiss.IndexFlatL2(DIMENSIONS)

flat_build_start = time.perf_counter()

flat_index.add(vectors)

flat_build_time = (
    time.perf_counter() - flat_build_start
) * 1000


def faiss_flat_search():

    distances, indices = flat_index.search(
        query,
        TOP_K
    )

    return indices[0]


flat_ids = faiss_flat_search()

flat_latency = measure_latency(
    faiss_flat_search
)

flat_recall = recall_at_k(
    numpy_ids,
    flat_ids
)

print("Nearest IDs:", flat_ids)
print(f"Index build time: {flat_build_time:.4f} ms")
print(f"FAISS Flat latency: {flat_latency:.4f} ms")
print(f"Recall@10: {flat_recall:.2f}")


# --------------------------------------------------
# 6. HNSW EXPERIMENTS
# --------------------------------------------------

print("\n========== HNSW EXPERIMENTS ==========")

M_VALUES = [8, 16, 32]

EF_SEARCH_VALUES = [16, 32, 64, 128, 256]

EF_CONSTRUCTION = 200

hnsw_results = []

for M in M_VALUES:

    print(f"\nBuilding HNSW index with M={M}")

    hnsw_index = faiss.IndexHNSWFlat(
        DIMENSIONS,
        M
    )

    hnsw_index.hnsw.efConstruction = EF_CONSTRUCTION

    build_start = time.perf_counter()

    hnsw_index.add(vectors)

    build_time = (
        time.perf_counter() - build_start
    ) * 1000

    print(f"Build time: {build_time:.2f} ms")

    for ef_search in EF_SEARCH_VALUES:

        hnsw_index.hnsw.efSearch = ef_search

        def hnsw_search():

            distances, indices = hnsw_index.search(
                query,
                TOP_K
            )

            return indices[0]

        approximate_ids = hnsw_search()

        latency = measure_latency(
            hnsw_search
        )

        recall = recall_at_k(
            numpy_ids,
            approximate_ids
        )

        result = {
            "M": M,
            "efSearch": ef_search,
            "efConstruction": EF_CONSTRUCTION,
            "build_time_ms": round(build_time, 4),
            "latency_ms": round(latency, 4),
            "recall_at_10": round(recall, 4)
        }

        hnsw_results.append(result)

        print(
            f"M={M:2d} | "
            f"efSearch={ef_search:3d} | "
            f"Latency={latency:8.4f} ms | "
            f"Recall@10={recall:.2f}"
        )


# --------------------------------------------------
# 7. SAVE RESULTS TO CSV
# --------------------------------------------------

csv_path = OUTPUT_DIR / "benchmark_results.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    fieldnames = [
        "M",
        "efSearch",
        "efConstruction",
        "build_time_ms",
        "latency_ms",
        "recall_at_10"
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(hnsw_results)

print("\nCSV saved:", csv_path)


# --------------------------------------------------
# 8. PLOT RECALL VS LATENCY
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for M in M_VALUES:

    results_for_M = [
        item for item in hnsw_results
        if item["M"] == M
    ]

    latencies = [
        item["latency_ms"]
        for item in results_for_M
    ]

    recalls = [
        item["recall_at_10"]
        for item in results_for_M
    ]

    plt.plot(
        latencies,
        recalls,
        marker="o",
        label=f"M={M}"
    )

    for item in results_for_M:

        plt.annotate(
            f'ef={item["efSearch"]}',
            (
                item["latency_ms"],
                item["recall_at_10"]
            ),
            fontsize=8
        )

plt.xlabel("Search Latency (milliseconds)")
plt.ylabel("Recall@10")

plt.title("HNSW Recall vs Search Latency")

plt.grid(True)
plt.legend()
plt.tight_layout()

recall_plot = OUTPUT_DIR / "recall_vs_latency.png"

plt.savefig(
    recall_plot,
    dpi=150
)

plt.close()

print("Recall plot saved:", recall_plot)


# --------------------------------------------------
# 9. EXACT VS APPROXIMATE COMPARISON
# --------------------------------------------------

best_hnsw = max(
    hnsw_results,
    key=lambda item: item["recall_at_10"]
)

comparison_names = [
    "NumPy Exact",
    "FAISS Flat",
    "HNSW Best Recall"
]

comparison_latencies = [
    numpy_latency,
    flat_latency,
    best_hnsw["latency_ms"]
]

plt.figure(figsize=(9, 5))

plt.bar(
    comparison_names,
    comparison_latencies
)

plt.ylabel("Search Latency (milliseconds)")
plt.title("Exact vs Approximate Search Latency")

plt.tight_layout()

comparison_plot = OUTPUT_DIR / "search_comparison.png"

plt.savefig(
    comparison_plot,
    dpi=150
)

plt.close()

print("Comparison plot saved:", comparison_plot)


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n========== FINAL COMPARISON ==========")

print(
    f"NumPy Exact: {numpy_latency:.4f} ms | "
    f"Recall@10: 1.00"
)

print(
    f"FAISS Flat: {flat_latency:.4f} ms | "
    f"Recall@10: {flat_recall:.2f}"
)

print(
    f"HNSW: {best_hnsw['latency_ms']:.4f} ms | "
    f"Recall@10: {best_hnsw['recall_at_10']:.2f}"
)

print(
    f"Best HNSW parameters: "
    f"M={best_hnsw['M']}, "
    f"efSearch={best_hnsw['efSearch']}"
)

print("\n========== TASK 1 COMPLETED ==========")
