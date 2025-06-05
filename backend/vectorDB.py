import json
import logging
import os
from typing import List, Dict, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ChromaVectorDB:
    def __init__(self, collection_name: str = "angelone_support", persist_dir: str = "./vector_db"):
        """Initialize ChromaDB vector database"""
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        # Create directory
        os.makedirs(persist_dir, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            logger.info(f"✅ Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            logger.info(f"✅ Created new collection: {collection_name}")

        logger.info(f"Collection has {self.collection.count()} documents")

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk.strip())

        return chunks

    def process_documents(self, documents: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
        """Process scraped documents into chunks with metadata"""
        all_texts = []
        all_ids = []
        all_metadata = []

        for doc_idx, doc in enumerate(documents):
            # Create chunks from the document content
            chunks = self.chunk_text(doc['content'])

            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"

                # Prepare metadata
                metadata = {
                    'source_url': doc['url'],
                    'title': doc['title'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'word_count': len(chunk.split()),
                    'doc_index': doc_idx
                }

                all_texts.append(chunk)
                all_ids.append(chunk_id)
                all_metadata.append(metadata)

        logger.info(f"Created {len(all_texts)} chunks from {len(documents)} documents")
        return all_texts, all_ids, all_metadata

    def build_index(self, documents: List[Dict]):
        """Build ChromaDB index from documents"""
        logger.info("Building ChromaDB index...")

        # Check if collection already has data
        if self.collection.count() > 0:
            logger.info("Collection already has data. Clearing...")
            # Clear existing data
            existing_ids = self.collection.get()['ids']
            if existing_ids:
                self.collection.delete(ids=existing_ids)

        # Process documents into chunks
        texts, ids, metadata = self.process_documents(documents)

        # Add documents to collection in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]

            self.collection.add(
                documents=batch_texts,
                ids=batch_ids,
                metadatas=batch_metadata
            )

            logger.info(f"Added batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

        logger.info(f"✅ ChromaDB index built with {self.collection.count()} chunks")

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict]:
        """Search for similar chunks"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k * 2,  # Get more results to filter
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            if results['documents'][0]:  # Check if we have results
                for i in range(len(results['documents'][0])):
                    # ChromaDB returns cosine distance, convert to similarity
                    distance = results['distances'][0][i]
                    similarity = 1 - distance if distance <= 1 else 1 / (1 + distance)

                    # Only include results above similarity threshold
                    if similarity >= min_similarity:
                        result = {
                            'text': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity_score': similarity,
                            'distance': distance
                        }
                        formatted_results.append(result)

                # Sort by similarity and limit to top_k
                formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
                formatted_results = formatted_results[:top_k]

            logger.info(f"Found {len(formatted_results)} results for query: '{query[:50]}...'")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            count = self.collection.count()
            return {
                "status": "ready",
                "total_chunks": count,
                "collection_name": self.collection_name,
                "persist_dir": self.persist_dir
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def delete_collection(self):
        """Delete the collection (for testing)"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")


# Test and build the vector database
def build_vector_db():
    """Build vector database from scraped data"""

    # Load scraped data
    with open('angelone_combined_data.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"📚 Loaded {len(documents)} documents")

    # Initialize vector DB
    vector_db = ChromaVectorDB()

    # Build index
    vector_db.build_index(documents)

    # Test search
    test_queries = [
        "How to add funds to my account?",
        "Where is germany located?",
        "How to withdraw money?",
        "IPO application process",
    ]

    print("\n🔍 TESTING SEARCH:")
    for query in test_queries:
        results = vector_db.search(query, top_k=3)
        print(f"\nQuery: '{query}'")

        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['metadata']['title']}")
                print(f"     Score: {result['similarity_score']:.3f}")
                print(f"     Preview: {result['text'][:100]}...")
        else:
            print("  No results found")

    # Print stats
    stats = vector_db.get_stats()
    print(f"\n📊 DATABASE STATS:")
    print(f"Status: {stats['status']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Collection: {stats['collection_name']}")

    return vector_db


if __name__ == "__main__":
    build_vector_db()

# if __name__ == "__main__":
#     # Load the database
#     vector_db = ChromaVectorDB()
#
#     # Test with exact queries
#     test_queries = [
#         "How do I withdraw money?",
#         "IPO application process",
#         "who is messi"
#     ]
#
#     for query in test_queries:
#         print(f"\nQuery: '{query}'")
#         results = vector_db.search(query, top_k=3, min_similarity=0.1)  # Very low threshold
#         for i, result in enumerate(results):
#             print(f"  {i+1}. Score: {result['similarity_score']:.3f} - {result['metadata']['title']}")
