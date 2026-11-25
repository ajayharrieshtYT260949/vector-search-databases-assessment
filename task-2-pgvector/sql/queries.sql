-- 1. Count all documents
SELECT COUNT(*) FROM documents;


-- 2. Exact cosine similarity search
-- %s represents a parameter supplied by Psycopg.

SELECT
    id,
    content,
    category,
    embedding <=> %s AS cosine_distance
FROM documents
ORDER BY embedding <=> %s
LIMIT 3;


-- 3. Filtered similarity search

SELECT
    id,
    content,
    category,
    embedding <=> %s AS cosine_distance
FROM documents
WHERE category = %s
ORDER BY embedding <=> %s
LIMIT 3;


-- 4. Create HNSW index for cosine similarity

CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
ON documents
USING hnsw (embedding vector_cosine_ops);


-- 5. Check indexes

SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'documents';


-- 6. Check query execution plan
-- The Python implementation supplies a real 384-dimensional vector.

EXPLAIN (ANALYZE, BUFFERS)
SELECT id, content
FROM documents
ORDER BY embedding <=> %s
LIMIT 3;