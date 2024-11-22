import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class VectorDB:
    def __init__(self, collection_name: str, dimension: int):
        """
        Initialize the vector database with the specified collection name and dimension.

        Args:
            collection_name (str): The name of the collection to use.
            dimension (int): The dimensionality of the vectors to be stored.
        """
        self.client = chromadb.Client(Settings())
        self.collection = self.client.create_collection(name=collection_name, dimension=dimension)

    def add_vectors(self, vectors, ids):
        """
        Add vectors to the database.

        Args:
            vectors (list): A list of vectors to add.
            ids (list): A list of IDs corresponding to the vectors.
        """
        self.collection.add(vectors=vectors, ids=ids)

    def search_vectors(self, query_vector, k=5):
        """
        Search for the k nearest vectors to the query vector.

        Args:
            query_vector (list): The query vector.
            k (int): The number of nearest neighbors to return.

        Returns:
            list: IDs of the nearest neighbors.
            list: Distances to the nearest neighbors.
        """
        results = self.collection.query(query_vector=query_vector, n_results=k)
        return results["ids"], results["distances"]


# Example usage
if __name__ == "__main__":
    # Initialize the vector database with a collection name and dimension
    vector_db = VectorDB(collection_name="example_collection", dimension=128)

    # Create some random vectors and IDs
    vectors = [[0.1] * 128, [0.2] * 128, [0.3] * 128]
    ids = ["vec1", "vec2", "vec3"]

    # Add vectors to the database
    vector_db.add_vectors(vectors, ids)

    # Create a query vector
    query_vector = [0.15] * 128

    # Search for the 5 nearest vectors
    nearest_ids, distances = vector_db.search_vectors(query_vector, k=5)
    print("Nearest IDs:", nearest_ids)
    print("Distances:", distances)
