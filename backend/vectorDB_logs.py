import json
import logging
import os
from typing import List, Dict, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaVectorDB:
    def __init__(self, collection_name: str = "angelone_support", persist_dir: str = "./vector_db"):
        """Initialize ChromaDB vector database"""
        logger.info("🚀 [INIT] Starting ChromaVectorDB initialization...")
        logger.info(f"🚀 [INIT] Collection name: {collection_name}")
        logger.info(f"🚀 [INIT] Persist directory: {persist_dir}")

        self.collection_name = collection_name
        self.persist_dir = persist_dir

        # Create directory
        logger.info("📁 [INIT] Creating persist directory...")
        os.makedirs(persist_dir, exist_ok=True)
        logger.info(f"📁 [INIT] ✅ Directory created/exists: {persist_dir}")

        # Initialize ChromaDB client with persistence
        logger.info("🔌 [INIT] Initializing ChromaDB PersistentClient...")
        self.client = chromadb.PersistentClient(path=persist_dir)
        logger.info("🔌 [INIT] ✅ ChromaDB client initialized successfully")

        # Initialize embedding model
        logger.info("🧠 [INIT] Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("🧠 [INIT] ✅ Embedding model loaded successfully")
        logger.info(
            f"🧠 [INIT] Model creates {self.embedding_model.get_sentence_embedding_dimension()}-dimensional vectors")

        # Get or create collection
        logger.info(f"🗃️  [INIT] Attempting to load existing collection '{collection_name}'...")
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            logger.info(f"🗃️  [INIT] ✅ Loaded existing collection: {collection_name}")
        except Exception as e:
            logger.info(f"🗃️  [INIT] Collection doesn't exist, creating new one. Error: {e}")
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            logger.info(f"🗃️  [INIT] ✅ Created new collection: {collection_name}")

        current_count = self.collection.count()
        logger.info(f"📊 [INIT] Collection currently has {current_count} documents")
        logger.info("🚀 [INIT] ✅ ChromaVectorDB initialization complete!")

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        logger.info(f"✂️  [CHUNK] Starting text chunking...")
        logger.info(f"✂️  [CHUNK] Input text length: {len(text)} characters")
        logger.info(f"✂️  [CHUNK] Chunk size: {chunk_size} words, Overlap: {overlap} words")

        words = text.split()
        logger.info(f"✂️  [CHUNK] Split into {len(words)} words")

        chunks = []
        step_size = chunk_size - overlap
        logger.info(f"✂️  [CHUNK] Step size (chunk_size - overlap): {step_size} words")

        for i in range(0, len(words), step_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, len(words))
            chunk = ' '.join(words[chunk_start:chunk_end])

            logger.info(f"✂️  [CHUNK] Processing chunk {len(chunks) + 1}: words {chunk_start}-{chunk_end}")
            logger.info(f"✂️  [CHUNK] Chunk length: {len(chunk)} characters, {len(chunk.split())} words")

            if len(chunk.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk.strip())
                logger.info(f"✂️  [CHUNK] ✅ Chunk {len(chunks)} saved (substantial content)")
                logger.info(f"✂️  [CHUNK] Preview: {chunk[:100]}...")
            else:
                logger.info(f"✂️  [CHUNK] ❌ Chunk skipped (too short: {len(chunk.strip())} chars)")

        logger.info(f"✂️  [CHUNK] ✅ Chunking complete! Created {len(chunks)} chunks from {len(words)} words")
        return chunks

    def process_documents(self, documents: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
        """Process scraped documents into chunks with metadata"""
        logger.info(f"📄 [PROCESS] Starting document processing...")
        logger.info(f"📄 [PROCESS] Total documents to process: {len(documents)}")

        all_texts = []
        all_ids = []
        all_metadata = []

        for doc_idx, doc in enumerate(documents):
            logger.info(f"📄 [PROCESS] Processing document {doc_idx + 1}/{len(documents)}")
            logger.info(f"📄 [PROCESS] Document title: {doc.get('title', 'No title')}")
            logger.info(f"📄 [PROCESS] Document URL: {doc.get('url', 'No URL')}")
            logger.info(f"📄 [PROCESS] Document content length: {len(doc.get('content', ''))} characters")

            # Create chunks from the document content
            logger.info(f"📄 [PROCESS] Chunking document {doc_idx + 1}...")
            chunks = self.chunk_text(doc['content'])
            logger.info(f"📄 [PROCESS] Document {doc_idx + 1} created {len(chunks)} chunks")

            for chunk_idx, chunk in enumerate(chunks):
                logger.info(f"📄 [PROCESS] Processing chunk {chunk_idx + 1}/{len(chunks)} from doc {doc_idx + 1}")

                # Create unique ID
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                logger.info(f"📄 [PROCESS] Generated chunk ID: {chunk_id}")

                # Prepare metadata
                metadata = {
                    'source_url': doc['url'],
                    'title': doc['title'],
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'word_count': len(chunk.split()),
                    'doc_index': doc_idx
                }
                logger.info(f"📄 [PROCESS] Chunk metadata: {metadata}")

                all_texts.append(chunk)
                all_ids.append(chunk_id)
                all_metadata.append(metadata)

                logger.info(f"📄 [PROCESS] ✅ Chunk {chunk_id} added to processing queue")
                logger.info(f"📄 [PROCESS] Current total chunks processed: {len(all_texts)}")

        logger.info(f"📄 [PROCESS] ✅ Document processing complete!")
        logger.info(f"📄 [PROCESS] Final stats: {len(all_texts)} chunks from {len(documents)} documents")
        logger.info(f"📄 [PROCESS] Average chunks per document: {len(all_texts) / len(documents):.1f}")
        return all_texts, all_ids, all_metadata

    def build_index(self, documents: List[Dict]):
        """Build ChromaDB index from documents"""
        logger.info("🏗️  [BUILD] Starting ChromaDB index build...")
        logger.info(f"🏗️  [BUILD] Input: {len(documents)} documents")

        # Check if collection already has data
        current_count = self.collection.count()
        logger.info(f"🏗️  [BUILD] Current collection count: {current_count}")

        if current_count > 0:
            logger.info("🏗️  [BUILD] Collection has existing data, clearing...")
            # Clear existing data
            existing_data = self.collection.get()
            existing_ids = existing_data['ids']
            logger.info(f"🏗️  [BUILD] Found {len(existing_ids)} existing items to delete")

            if existing_ids:
                logger.info("🏗️  [BUILD] Deleting existing data...")
                self.collection.delete(ids=existing_ids)
                logger.info("🏗️  [BUILD] ✅ Existing data cleared")
            else:
                logger.info("🏗️  [BUILD] No existing IDs found to delete")

        # Process documents into chunks
        logger.info("🏗️  [BUILD] Processing documents into chunks...")
        texts, ids, metadata = self.process_documents(documents)
        logger.info(f"🏗️  [BUILD] Ready to insert {len(texts)} chunks into ChromaDB")

        # Add documents to collection in batches
        batch_size = 100
        total_batches = (len(texts) - 1) // batch_size + 1
        logger.info(
            f"🏗️  [BUILD] Batch insertion starting: {batch_size} items per batch, {total_batches} total batches")

        for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
            logger.info(f"🏗️  [BUILD] Processing batch {batch_num}/{total_batches}")

            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]

            logger.info(f"🏗️  [BUILD] Batch {batch_num} contains {len(batch_texts)} items")
            logger.info(f"🏗️  [BUILD] Batch range: items {i} to {min(i + batch_size - 1, len(texts) - 1)}")
            logger.info(f"🏗️  [BUILD] Sample IDs in this batch: {batch_ids[:3]}...")

            logger.info(f"🏗️  [BUILD] Inserting batch {batch_num} into ChromaDB...")
            self.collection.add(
                documents=batch_texts,
                ids=batch_ids,
                metadatas=batch_metadata
            )

            # Verify insertion
            current_count_after_batch = self.collection.count()
            logger.info(f"🏗️  [BUILD] ✅ Batch {batch_num} inserted successfully")
            logger.info(f"🏗️  [BUILD] Collection count after batch {batch_num}: {current_count_after_batch}")

            # Sample a few items to verify they were inserted correctly
            if batch_num == 1:  # Only do this for first batch to avoid spam
                logger.info("🏗️  [BUILD] Verifying insertion by fetching sample items...")
                sample_fetch = self.collection.get(ids=batch_ids[:2], include=['documents', 'metadatas'])
                for idx, (sample_id, sample_doc, sample_meta) in enumerate(zip(
                        sample_fetch['ids'], sample_fetch['documents'], sample_fetch['metadatas']
                )):
                    logger.info(f"🏗️  [BUILD] Sample item {idx + 1}:")
                    logger.info(f"🏗️  [BUILD]   ID: {sample_id}")
                    logger.info(f"🏗️  [BUILD]   Metadata: {sample_meta}")
                    logger.info(f"🏗️  [BUILD]   Content preview: {sample_doc[:100]}...")

        final_count = self.collection.count()
        logger.info(f"🏗️  [BUILD] ✅ Index build complete!")
        logger.info(f"🏗️  [BUILD] Final collection count: {final_count}")
        logger.info(f"🏗️  [BUILD] Expected count: {len(texts)}")

        if final_count == len(texts):
            logger.info("🏗️  [BUILD] ✅ Count verification passed - all items inserted successfully")
        else:
            logger.warning(f"🏗️  [BUILD] ⚠️  Count mismatch! Expected {len(texts)}, got {final_count}")

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict]:
        """Search for similar chunks"""
        logger.info(f"🔍 [SEARCH] Starting search...")
        logger.info(f"🔍 [SEARCH] Query: '{query}'")
        logger.info(f"🔍 [SEARCH] Top K: {top_k}")
        logger.info(f"🔍 [SEARCH] Min similarity threshold: {min_similarity}")

        try:
            logger.info("🔍 [SEARCH] Executing ChromaDB query...")
            search_results_count = top_k * 2
            logger.info(f"🔍 [SEARCH] Requesting {search_results_count} results for filtering")

            results = self.collection.query(
                query_texts=[query],
                n_results=search_results_count,
                include=['documents', 'metadatas', 'distances']
            )

            logger.info("🔍 [SEARCH] ✅ ChromaDB query completed")
            logger.info(
                f"🔍 [SEARCH] Raw results count: {len(results['documents'][0]) if results['documents'][0] else 0}")

            # Format results
            formatted_results = []
            if results['documents'][0]:  # Check if we have results
                logger.info("🔍 [SEARCH] Processing and filtering results...")

                for i in range(len(results['documents'][0])):
                    # ChromaDB returns cosine distance, convert to similarity
                    distance = results['distances'][0][i]
                    similarity = 1 - distance if distance <= 1 else 1 / (1 + distance)

                    logger.info(f"🔍 [SEARCH] Result {i + 1}:")
                    logger.info(f"🔍 [SEARCH]   Distance: {distance:.4f}")
                    logger.info(f"🔍 [SEARCH]   Similarity: {similarity:.4f}")
                    logger.info(f"🔍 [SEARCH]   Title: {results['metadatas'][0][i].get('title', 'No title')}")
                    logger.info(f"🔍 [SEARCH]   Content preview: {results['documents'][0][i][:100]}...")

                    # Only include results above similarity threshold
                    if similarity >= min_similarity:
                        result = {
                            'text': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity_score': similarity,
                            'distance': distance
                        }
                        formatted_results.append(result)
                        logger.info(
                            f"🔍 [SEARCH]   ✅ Result {i + 1} passed threshold (similarity: {similarity:.4f} >= {min_similarity})")
                    else:
                        logger.info(
                            f"🔍 [SEARCH]   ❌ Result {i + 1} filtered out (similarity: {similarity:.4f} < {min_similarity})")

                # Sort by similarity and limit to top_k
                logger.info(f"🔍 [SEARCH] Sorting {len(formatted_results)} results by similarity...")
                formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)

                original_count = len(formatted_results)
                formatted_results = formatted_results[:top_k]
                final_count = len(formatted_results)

                logger.info(f"🔍 [SEARCH] Limited results from {original_count} to top {final_count}")

                # Log final results summary
                logger.info("🔍 [SEARCH] Final results summary:")
                for idx, result in enumerate(formatted_results, 1):
                    logger.info(
                        f"🔍 [SEARCH]   {idx}. Score: {result['similarity_score']:.4f} - {result['metadata'].get('title', 'No title')}")
            else:
                logger.info("🔍 [SEARCH] No results returned from ChromaDB")

            logger.info(f"🔍 [SEARCH] ✅ Search complete! Returning {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"🔍 [SEARCH] ❌ Error during search: {e}")
            logger.error(f"🔍 [SEARCH] Query that failed: '{query}'")
            return []

    def get_stats(self) -> Dict:
        """Get database statistics"""
        logger.info("📊 [STATS] Generating database statistics...")
        try:
            count = self.collection.count()
            stats = {
                "status": "ready",
                "total_chunks": count,
                "collection_name": self.collection_name,
                "persist_dir": self.persist_dir
            }
            logger.info(f"📊 [STATS] ✅ Stats generated successfully: {stats}")
            return stats
        except Exception as e:
            error_stats = {
                "status": "error",
                "error": str(e)
            }
            logger.error(f"📊 [STATS] ❌ Error generating stats: {error_stats}")
            return error_stats

    def delete_collection(self):
        """Delete the collection (for testing)"""
        logger.info(f"🗑️  [DELETE] Attempting to delete collection: {self.collection_name}")
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"🗑️  [DELETE] ✅ Collection deleted successfully: {self.collection_name}")
        except Exception as e:
            logger.warning(f"🗑️  [DELETE] ⚠️  Could not delete collection: {e}")


# Test and build the vector database
def build_vector_db():
    """Build vector database from scraped data"""
    logger.info("🚀 [MAIN] Starting build_vector_db function...")

    # Load scraped data
    json_file = 'angelone_support_data.json'
    logger.info(f"📖 [MAIN] Loading scraped data from {json_file}...")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        logger.info(f"📖 [MAIN] ✅ Successfully loaded {len(documents)} documents")

        # Log sample document structure
        if documents:
            sample_doc = documents[0]
            logger.info("📖 [MAIN] Sample document structure:")
            logger.info(f"📖 [MAIN]   Keys: {list(sample_doc.keys())}")
            logger.info(f"📖 [MAIN]   Title: {sample_doc.get('title', 'No title')}")
            logger.info(f"📖 [MAIN]   URL: {sample_doc.get('url', 'No URL')}")
            logger.info(f"📖 [MAIN]   Content length: {len(sample_doc.get('content', ''))} chars")
    except FileNotFoundError:
        logger.error(f"📖 [MAIN] ❌ File not found: {json_file}")
        return None
    except Exception as e:
        logger.error(f"📖 [MAIN] ❌ Error loading file: {e}")
        return None

    # Initialize vector DB
    logger.info("🔧 [MAIN] Initializing ChromaVectorDB...")
    vector_db = ChromaVectorDB()

    # Build index
    logger.info("🏗️  [MAIN] Building vector database index...")
    vector_db.build_index(documents)

    # Test search
    test_queries = [
        "How to add funds to my account?",
        "How to operate a brain surgery?"
    ]

    logger.info("🧪 [MAIN] Starting search tests...")
    print("\n🔍 TESTING SEARCH:")
    for query_idx, query in enumerate(test_queries, 1):
        logger.info(f"🧪 [MAIN] Test query {query_idx}/{len(test_queries)}: '{query}'")
        results = vector_db.search(query, top_k=3)
        print(f"\nQuery: '{query}'")

        if results:
            logger.info(f"🧪 [MAIN] Query '{query}' returned {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['metadata']['title']}")
                print(f"     Score: {result['similarity_score']:.3f}")
                print(f"     Preview: {result['text'][:100]}...")
        else:
            logger.info(f"🧪 [MAIN] Query '{query}' returned no results")
            print("  No results found")

    # Print stats
    logger.info("📊 [MAIN] Generating final database statistics...")
    stats = vector_db.get_stats()
    print(f"\n📊 DATABASE STATS:")
    print(f"Status: {stats['status']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Collection: {stats['collection_name']}")

    logger.info("🚀 [MAIN] ✅ build_vector_db function completed successfully")
    return vector_db


if __name__ == "__main__":
    logger.info("🌟 [ENTRY] Script started - running build_vector_db")
    result = build_vector_db()
    if result:
        logger.info("🌟 [ENTRY] ✅ Script completed successfully")
    else:
        logger.error("🌟 [ENTRY] ❌ Script failed")
