"""
Data Processor for Farmer Subsidy Information

This script processes CSV data containing agricultural subsidy information and loads it into Qdrant.
It creates embeddings using Google's Gemini API and stores them in a Qdrant collection for efficient retrieval.
"""

import os
import pandas as pd
from dotenv import load_dotenv
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http import models
import google.generativeai as genai

# Load environment variables
load_dotenv()
print("DEBUG QDRANT_URL =", os.getenv("QDRANT_URL"))
print("DEBUG QDRANT_API_KEY =", os.getenv("QDRANT_API_KEY"))

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Import the embedding service
from google.generativeai import embed_content

def get_embedding(text):
    """Generate embeddings using Gemini embedding model."""
    # Use the embed_content function from version 0.8.5
    response = embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    
    # Get the embedding values
    return response["embedding"]


def create_qdrant_client(use_memory=True, host=None, port=None, url=None, api_key=None):
    """Create and return a Qdrant client."""
    if use_memory:
        return QdrantClient(":memory:")

    # Use env variables if not explicitly passed
    url = url or os.getenv("QDRANT_URL")
    api_key = api_key or os.getenv("QDRANT_API_KEY")

    if url:
        return QdrantClient(url=url, api_key=api_key)
    elif host and port:
        return QdrantClient(host=host, port=port)
    else:
        raise ValueError("Invalid Qdrant configuration: URL or host:port required")

def create_collection(client, collection_name="farmer_subsidies", vector_size=768):
    """Create a collection in Qdrant for storing subsidy embeddings."""
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists")
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection '{collection_name}'")

def process_csv_file(file_path):
    """Process the CSV file and return a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)

    print(f"Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns\n")
    print("Column names:")
    for col in df.columns:
        print(f"- {col}")

    print("\nSample data (first 2 rows):")
    print(df.head(2))
    return df

def process_row(row):
    """Convert a row into a combined text document for embedding."""
    document = ""
    for col, value in row.items():
        if pd.notna(value):
            document += f"{col}: {value}\n"
    return document

def upload_data_to_qdrant(client, df, collection_name="farmer_subsidies", batch_size=100):
    """Upload DataFrame content to Qdrant."""
    points = []
    total_processed = 0

    for idx, row in df.iterrows():
        document = process_row(row)
        embedding = get_embedding(document)

        payload = {col: str(row[col]) if pd.notna(row[col]) else "" for col in df.columns}
        payload["full_text"] = document

        points.append(
            models.PointStruct(
                id=idx,
                vector=embedding,
                payload=payload
            )
        )

        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            total_processed += len(points)
            print(f"Uploaded {total_processed}/{df.shape[0]} records to Qdrant")
            points = []

    if points:
        client.upsert(collection_name=collection_name, points=points)
        total_processed += len(points)
        print(f"Uploaded {total_processed}/{df.shape[0]} records to Qdrant")

    print(f"✅ All {total_processed} records uploaded to Qdrant successfully.")

def main():
    """Main execution entry point."""
    parser = argparse.ArgumentParser(description="Process CSV data and upload to Qdrant")
    parser.add_argument("csv_file", help="D:\Sem6\AI\website-scrapper\scheme_data.csv")
    parser.add_argument("--collection", default="farmer_subsidies", help="farmer_subsidies")
    parser.add_argument("--memory", action="store_true", help="Use in-memory Qdrant (for testing)")
    parser.add_argument("--host", default="localhost", help="Qdrant server host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--url", help="Qdrant Cloud URL (overrides env var)")
    parser.add_argument("--api-key", help="Qdrant API Key (overrides env var)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for uploading to Qdrant")

    args = parser.parse_args()

    client = create_qdrant_client(
        use_memory=args.memory,
        host=args.host,
        port=args.port,
        url=args.url,
        api_key=args.api_key
    )

    create_collection(client, args.collection)
    df = process_csv_file(args.csv_file)
    upload_data_to_qdrant(client, df, args.collection, args.batch_size)
    print("🎉 Data processing complete!")

if __name__ == "__main__":
    main()