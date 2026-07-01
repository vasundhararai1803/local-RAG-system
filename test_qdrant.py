import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url="http://localhost:6333")
collection_name = uuid.uuid4().hex

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}
)

info = client.get_collection(collection_name)
print("Sparse vectors config:", info.config.params.sparse_vectors_config)
