from datetime import datetime, timedelta, timezone
import logging
from typing import List, Optional


import vertexai
from google.cloud import storage
from google.cloud import firestore 
from google.cloud.aiplatform.matching_engine import MatchingEngineIndex
from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint
from vertexai.language_models import TextEmbeddingModel
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from config.settings import get_settings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_firestore import FirestoreVectorStore


settings = get_settings()  

PROJECT_ID = settings.GCP_PROJECT_ID

TTL_DAYS = 1



embedding = VertexAIEmbeddings(
    model_name="text-multilingual-embedding-002",
    project=PROJECT_ID
)

vector_store = FirestoreVectorStore(
    collection="query_cache",
    embedding_service=embedding
)


def save_query(query_text: str, sql_query: str) -> str:
    """
    Saves a new query and its corresponding SQL to Firestore metadata,
    and directly adds its embedding to the deployed streaming Vector Search Index.
    Includes a TTL expiration timestamp.

    Args:
        query_text: The natural language query.
        sql_query: The SQL query corresponding to the natural language query.

    Returns:
        The unique datapoint ID generated for this entry.
    """
    expiration_time = datetime.utcnow() + timedelta(days=TTL_DAYS)


    ids = vector_store.add_texts(
        texts=[query_text],
        metadatas=[{"sql": sql_query}]
    )
    doc_id = ids[0]


    db = firestore.Client()
    doc_ref = db.collection("query_cache").document(doc_id)
    doc_ref.update({"expiration": expiration_time})

    return doc_id


def retrieve_query(query_text: str, num_results: int = 1, threshold: float = 0.4) -> Optional[List[str]]:
    """
    Retrieves SQL queries from Firestore whose natural language embeddings are similar
    to the input query, within a distance threshold.

    Args:
        query_text: The natural language query to search for.
        num_results: Max number of similar queries to retrieve.
        threshold: Max distance allowed for a match (lower = more similar).

    Returns:
        A list of SQL strings (max one by default) or None if no good match found.
    """
    try:
        query_vector = embedding.embed_query(query_text)  


        db = firestore.Client(project=PROJECT_ID)
        collection = db.collection("query_cache")

        docs = list(db.collection("query_cache").find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_vector),
            distance_measure=DistanceMeasure.COSINE,
            limit=num_results,
            distance_threshold=threshold
        ).stream())


        results = []
        if not docs:
            print("No documents found in Firestore for the given query.")
            logging.info("No documents found in Firestore for the given query.")
            return None
        else:

            data = docs[0].to_dict()
            metadata = data.get("metadata")
            return metadata.get("sql", None)

    except Exception as e:
        print(f"Error in retrieve_query: {e}")
        logging.exception("retrieve_query failed")
        return None
