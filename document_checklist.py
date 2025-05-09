"""
Document Checklist Generator for Farmer Subsidy Applications

This module extends the existing Farmer Subsidy RAG Chatbot to include
a document checklist generator feature. It helps farmers prepare all
necessary paperwork before visiting government offices.
"""

import os
import pandas as pd
import streamlit as st
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the translator
translator = GoogleTranslator(source='auto', target='hi')


# Initialize Qdrant client - using the same client as in the main application
qdrant_client = QdrantClient(
    url="https://ae245ab4-d048-42a9-bff3-f32bb853dd3f.eu-west-2-0.aws.cloud.qdrant.io", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.6Uho4S7-1Yjne4NcJZdLOioFDy2nkfpkPINi39rjzPw"
)

# Set up Gemini model for chat
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Function from original code to get embeddings
def get_embedding(text):
    """Get embedding from Google's text-embedding model."""
    try:
        embedding_result = genai.embed_content(
            model="embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        
        if hasattr(embedding_result, 'embedding'):
            return embedding_result.embedding
        elif isinstance(embedding_result, dict) and 'embedding' in embedding_result:
            return embedding_result['embedding']
        else:
            print(f"Unexpected embedding result structure: {embedding_result}")
            return None
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# Create a new collection for document requirements if it doesn't exist
def create_document_requirements_collection(collection_name="document_requirements", vector_size=768):
    """Create a collection in Qdrant for storing document requirements."""
    try:
        qdrant_client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists")
    except Exception:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection {collection_name}")

# Process and upload document requirements data
def upload_document_requirements(df, collection_name="document_requirements"):
    """Upload document requirements data to Qdrant."""
    points = []
    
    for idx, row in df.iterrows():
        # Create a structured document from the row data
        document = f"Scheme: {row.get('scheme_name', '')}\n"
        document += f"Document: {row.get('document_name', '')}\n"
        document += f"Required: {row.get('required', 'Yes')}\n"
        document += f"Difficulty: {row.get('difficulty', 'Medium')}\n"
        document += f"Processing Time: {row.get('processing_time', 'Moderate')}\n"
        document += f"Cost: {row.get('cost', 'Free')}\n"
        document += f"Validity: {row.get('validity', 'No expiry')}\n"
        document += f"Category: {row.get('category', 'General')}\n"
        document += f"Notes: {row.get('notes', '')}\n"
        
        # Generate embedding for the document
        embedding = get_embedding(document)
        
        # Skip if embedding generation failed
        if embedding is None:
            print(f"Skipping row {idx} due to embedding generation failure")
            continue
        
        # Create Qdrant point
        points.append(
            models.PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "scheme_name": row.get("scheme_name", ""),
                    "document_name": row.get("document_name", ""),
                    "required": row.get("required", "Yes"),
                    "difficulty": row.get("difficulty", "Medium"),
                    "processing_time": row.get("processing_time", "Moderate"),
                    "cost": row.get("cost", "Free"),
                    "validity": row.get("validity", "No expiry"),
                    "category": row.get("category", "General"),
                    "notes": row.get("notes", ""),
                    "full_text": document
                }
            )
        )
        
        # Upload in batches of 100 to avoid memory issues
        if len(points) >= 100:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            points = []
    
    # Upload any remaining points
    if points:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    print(f"Uploaded {df.shape[0]} document requirements to Qdrant")

# Create a sample document requirements dataframe
def create_sample_document_requirements():
    """Create a sample dataframe with document requirements data."""
    data = [
        # PM-KISAN documents
        {
            "scheme_name": "PM-KISAN",
            "document_name": "Aadhaar Card",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "No expiry",
            "category": "Identity",
            "notes": "Must be linked with mobile number"
        },
        {
            "scheme_name": "PM-KISAN",
            "document_name": "Land records (7/12 extract, khata/khatauni, khasra)",
            "required": "Mandatory",
            "difficulty": "Medium",
            "processing_time": "Moderate",
            "cost": "Low",
            "validity": "Recent (within 6 months)",
            "category": "Land",
            "notes": "Get from local revenue office (Tehsil/Taluka)"
        },
        {
            "scheme_name": "PM-KISAN",
            "document_name": "Bank account passbook with IFSC code",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "Active account",
            "category": "Financial",
            "notes": "Account must be in applicant's name"
        },
        {
            "scheme_name": "PM-KISAN",
            "document_name": "Recent passport-sized photograph",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Low",
            "validity": "Recent (within 6 months)",
            "category": "Identity",
            "notes": "White background preferred"
        },
        {
            "scheme_name": "PM-KISAN",
            "document_name": "Self-declaration form",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "For current application",
            "category": "Declaration",
            "notes": "Declaration of eligibility"
        },
        
        # Kisan Credit Card documents
        {
            "scheme_name": "Kisan Credit Card",
            "document_name": "Aadhaar Card",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "No expiry",
            "category": "Identity",
            "notes": "Must be linked with mobile number"
        },
        {
            "scheme_name": "Kisan Credit Card",
            "document_name": "Land ownership documents or lease agreement",
            "required": "Mandatory",
            "difficulty": "Hard",
            "processing_time": "Long",
            "cost": "Moderate",
            "validity": "Recent (within 6 months)",
            "category": "Land",
            "notes": "For leased land, agreement must be registered"
        },
        {
            "scheme_name": "Kisan Credit Card",
            "document_name": "Passport-sized photographs",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Low",
            "validity": "Recent (within 6 months)",
            "category": "Identity",
            "notes": "2-3 copies required"
        },
        {
            "scheme_name": "Kisan Credit Card",
            "document_name": "Proof of residence",
            "required": "Mandatory",
            "difficulty": "Medium",
            "processing_time": "Moderate",
            "cost": "Low",
            "validity": "Recent (within 3 months)",
            "category": "Identity",
            "notes": "Utility bill/voter ID/ration card"
        },
        {
            "scheme_name": "Kisan Credit Card",
            "document_name": "Income certificate or self-declaration",
            "required": "Mandatory",
            "difficulty": "Medium",
            "processing_time": "Moderate",
            "cost": "Low",
            "validity": "Recent (within 6 months)",
            "category": "Financial",
            "notes": "From local revenue office"
        },
        {
            "scheme_name": "Kisan Credit Card",
            "document_name": "Crop pattern details",
            "required": "Mandatory",
            "difficulty": "Medium",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "For previous 2 seasons",
            "category": "Farm",
            "notes": "Details of crops grown in last 2 seasons"
        },
        {
            "scheme_name": "Kisan Credit Card",
            "document_name": "Existing loan details",
            "required": "Conditional",
            "difficulty": "Medium",
            "processing_time": "Moderate",
            "cost": "Free",
            "validity": "Recent (within 3 months)",
            "category": "Financial",
            "notes": "If any existing agricultural loans"
        },
        
        # PM Fasal Bima Yojana documents
        {
            "scheme_name": "PM Fasal Bima Yojana",
            "document_name": "Aadhaar Card",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "No expiry",
            "category": "Identity",
            "notes": "Must be linked with mobile number"
        },
        {
            "scheme_name": "PM Fasal Bima Yojana",
            "document_name": "Land records with crop sowing details",
            "required": "Mandatory",
            "difficulty": "Hard",
            "processing_time": "Long",
            "cost": "Moderate",
            "validity": "Current season",
            "category": "Land",
            "notes": "Must show current crop details"
        },
        {
            "scheme_name": "PM Fasal Bima Yojana",
            "document_name": "Bank account details",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "Active account",
            "category": "Financial",
            "notes": "For premium deduction and claim disbursement"
        },
        {
            "scheme_name": "PM Fasal Bima Yojana",
            "document_name": "Previous crop loss records",
            "required": "Conditional",
            "difficulty": "Medium",
            "processing_time": "Moderate",
            "cost": "Low",
            "validity": "Previous seasons",
            "category": "Farm",
            "notes": "If claiming based on previous losses"
        },
        
        # Soil Health Card Scheme documents
        {
            "scheme_name": "Soil Health Card Scheme",
            "document_name": "Aadhaar Card",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "No expiry",
            "category": "Identity",
            "notes": "For identification"
        },
        {
            "scheme_name": "Soil Health Card Scheme",
            "document_name": "Land location details",
            "required": "Mandatory",
            "difficulty": "Medium",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "Current",
            "category": "Land",
            "notes": "Survey number and location map"
        },
        {
            "scheme_name": "Soil Health Card Scheme",
            "document_name": "Previous soil test reports",
            "required": "Optional",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "Previous tests",
            "category": "Farm",
            "notes": "If available"
        },
        
        # Per Drop More Crop documents
        {
            "scheme_name": "Per Drop More Crop",
            "document_name": "Aadhaar Card",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "No expiry",
            "category": "Identity",
            "notes": "For identification"
        },
        {
            "scheme_name": "Per Drop More Crop",
            "document_name": "Land documents showing ownership",
            "required": "Mandatory",
            "difficulty": "Hard",
            "processing_time": "Long",
            "cost": "Moderate",
            "validity": "Recent (within 6 months)",
            "category": "Land",
            "notes": "Must be in applicant's name"
        },
        {
            "scheme_name": "Per Drop More Crop",
            "document_name": "Bank account details",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "Active account",
            "category": "Financial",
            "notes": "For subsidy transfer"
        },
        {
            "scheme_name": "Per Drop More Crop",
            "document_name": "Water source proof",
            "required": "Mandatory",
            "difficulty": "Medium",
            "processing_time": "Moderate",
            "cost": "Low",
            "validity": "Current",
            "category": "Farm",
            "notes": "Certificate from irrigation department"
        },
        {
            "scheme_name": "Per Drop More Crop",
            "document_name": "Quotation from authorized equipment vendor",
            "required": "Mandatory",
            "difficulty": "Medium",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "Valid for 3 months",
            "category": "Financial",
            "notes": "Must be from authorized dealer"
        },
        {
            "scheme_name": "Per Drop More Crop",
            "document_name": "Field photographs",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "Recent (within 1 month)",
            "category": "Farm",
            "notes": "Showing current farm condition"
        },
        
        # National Mission for Sustainable Agriculture documents
        {
            "scheme_name": "National Mission for Sustainable Agriculture",
            "document_name": "Aadhaar Card",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "No expiry",
            "category": "Identity",
            "notes": "For identification"
        },
        {
            "scheme_name": "National Mission for Sustainable Agriculture",
            "document_name": "Land documents",
            "required": "Mandatory",
            "difficulty": "Hard",
            "processing_time": "Long",
            "cost": "Moderate",
            "validity": "Recent (within 6 months)",
            "category": "Land",
            "notes": "Must be in applicant's name"
        },
        {
            "scheme_name": "National Mission for Sustainable Agriculture",
            "document_name": "Bank account details",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "Active account",
            "category": "Financial",
            "notes": "For subsidy transfer"
        },
        {
            "scheme_name": "National Mission for Sustainable Agriculture",
            "document_name": "Proposal for farming techniques",
            "required": "Mandatory",
            "difficulty": "Hard",
            "processing_time": "Moderate",
            "cost": "Free",
            "validity": "For current application",
            "category": "Farm",
            "notes": "Detailed proposal of sustainable practices"
        },
        {
            "scheme_name": "National Mission for Sustainable Agriculture",
            "document_name": "Photographs of current farm condition",
            "required": "Mandatory",
            "difficulty": "Easy",
            "processing_time": "Quick",
            "cost": "Free",
            "validity": "Recent (within 1 month)",
            "category": "Farm",
            "notes": "Multiple angles of farm"
        },
        {
            "scheme_name": "National Mission for Sustainable Agriculture",
            "document_name": "Income declaration",
            "required": "Mandatory",
            "difficulty": "Medium",
            "processing_time": "Moderate",
            "cost": "Low",
            "validity": "For current application",
            "category": "Financial",
            "notes": "Self-declaration of income"
        }
    ]
    
    return pd.DataFrame(data)

# Function to search for document requirements by scheme
def search_document_requirements(scheme_name, collection_name="document_requirements", limit=30):
    """Search for document requirements by scheme name."""
    query = f"Scheme: {scheme_name}"
    query_embedding = get_embedding(query)
    
    if query_embedding is None:
        print("Failed to generate embedding for search query")
        return []
    
    try:
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Filter results to only include the specified scheme
        filtered_results = [
            result for result in search_results 
            if result.payload.get("scheme_name", "").lower() == scheme_name.lower()
        ]
        
        return filtered_results
    except Exception as e:
        print(f"Error searching document requirements: {e}")
        return []

# Function to generate a document checklist based on selected schemes
def generate_document_checklist(selected_schemes, lang="en"):
    """Generate a document checklist based on selected schemes."""
    all_documents = {}
    
    # Get document requirements for each selected scheme
    for scheme in selected_schemes:
        scheme_documents = search_document_requirements(scheme)
        
        for doc in scheme_documents:
            doc_name = doc.payload.get("document_name", "")
            # If document already exists, just add this scheme to its required_for list
            if doc_name in all_documents:
                all_documents[doc_name]["required_for"].append(scheme)
            else:
                all_documents[doc_name] = {
                    "name": doc_name,
                    "required": doc.payload.get("required", "Yes"),
                    "difficulty": doc.payload.get("difficulty", "Medium"),
                    "processing_time": doc.payload.get("processing_time", "Moderate"),
                    "cost": doc.payload.get("cost", "Free"),
                    "validity": doc.payload.get("validity", "No expiry"),
                    "category": doc.payload.get("category", "General"),
                    "notes": doc.payload.get("notes", ""),
                    "required_for": [scheme]
                }
    
    # Organize documents by category
    organized_documents = {
        "Identity": [],
        "Land": [],
        "Financial": [],
        "Farm": [],
        "Declaration": [],
        "General": []
    }
    
    for doc_name, doc_info in all_documents.items():
        category = doc_info["category"]
        if category in organized_documents:
            organized_documents[category].append(doc_info)
        else:
            organized_documents["General"].append(doc_info)
    
    # Sort documents within each category by difficulty
    difficulty_order = {"Easy": 0, "Medium": 1, "Hard": 2}
    for category in organized_documents:
        organized_documents[category].sort(
            key=lambda x: (difficulty_order.get(x["difficulty"], 1))
        )
    
    return organized_documents

# Function to format the document checklist for display
def format_document_checklist(organized_documents, selected_schemes):
    """Format the document checklist for display in Streamlit."""
    # Start with a header
    checklist = f"# Document Checklist for Selected Schemes\n\n"
    checklist += f"**Selected Schemes:** {', '.join(selected_schemes)}\n\n"
    
    # Add a summary of what to expect
    total_documents = sum(len(docs) for docs in organized_documents.values())
    easy_docs = sum(1 for cat in organized_documents.values() for doc in cat if doc["difficulty"] == "Easy")
    medium_docs = sum(1 for cat in organized_documents.values() for doc in cat if doc["difficulty"] == "Medium")
    hard_docs = sum(1 for cat in organized_documents.values() for doc in cat if doc["difficulty"] == "Hard")
    
    checklist += f"**Summary:** You need {total_documents} documents total:\n"
    checklist += f"- {easy_docs} easy to obtain\n"
    checklist += f"- {medium_docs} medium difficulty\n"
    checklist += f"- {hard_docs} harder to obtain\n\n"
    
    # Add a note about planning ahead
    checklist += "**Important:** Start collecting the more difficult documents first!\n\n"
    
    # Add each category of documents
    for category, documents in organized_documents.items():
        if documents:  # Only include categories that have documents
            checklist += f"## {category} Documents\n\n"
            
            for i, doc in enumerate(documents, 1):
                # Format each document with all details
                checklist += f"### {i}. {doc['name']}\n"
                checklist += f"- **Required for:** {', '.join(doc['required_for'])}\n"
                checklist += f"- **Requirement:** {doc['required']}\n"
                
                # Use emoji indicators for difficulty
                difficulty_emoji = "🟢" if doc["difficulty"] == "Easy" else "🟡" if doc["difficulty"] == "Medium" else "🔴"
                checklist += f"- **Difficulty:** {difficulty_emoji} {doc['difficulty']}\n"
                
                time_emoji = "⚡" if doc["processing_time"] == "Quick" else "⏱️" if doc["processing_time"] == "Moderate" else "⏳"
                checklist += f"- **Processing time:** {time_emoji} {doc['processing_time']}\n"
                
                cost_emoji = "💰" if doc["cost"] != "Free" else "🆓"
                checklist += f"- **Cost:** {cost_emoji} {doc['cost']}\n"
                
                checklist += f"- **Validity:** {doc['validity']}\n"
                
                if doc["notes"]:
                    checklist += f"- **Notes:** {doc['notes']}\n"
                
                checklist += "\n"
    
    # Add tips at the end
    checklist += "## Tips for Document Collection\n\n"
    checklist += "1. **Start early:** Begin with the most difficult documents first\n"
    checklist += "2. **Make copies:** Keep multiple photocopies of all documents\n"
    checklist += "3. **Digital backup:** Scan all documents and store them on your phone\n"
    checklist += "4. **Check validity:** Ensure all documents with expiry dates are current\n"
    checklist += "5. **Organize properly:** Keep documents sorted by category in a folder\n"
    
    return checklist

# Function to add document checklist feature to Streamlit app
def add_document_checklist_to_app():
    """Add document checklist generator to the Streamlit app."""
    st.title("📑 Document Checklist Generator")
    st.subheader("Find out which documents you need for agricultural subsidies")
    
    # Create a multiselect for choosing schemes
    available_schemes = [
        "PM-KISAN",
        "Kisan Credit Card",
        "PM Fasal Bima Yojana",
        "Soil Health Card Scheme",
        "Per Drop More Crop",
        "National Mission for Sustainable Agriculture"
    ]
    
    selected_schemes = st.multiselect(
        "Select schemes you're interested in applying for:",
        available_schemes,
        default=["PM-KISAN"]
    )
    
    # Add a button to generate the checklist
    if st.button("Generate Document Checklist"):
        if selected_schemes:
            with st.spinner("Generating your personalized document checklist..."):
                # Generate the checklist
                organized_documents = generate_document_checklist(selected_schemes)
                checklist = format_document_checklist(organized_documents, selected_schemes)
                
                # Display the checklist
                st.markdown(checklist)
                
                # Add a download button for the checklist
                st.download_button(
                    label="Download Checklist as Text",
                    data=checklist,
                    file_name="my_subsidy_document_checklist.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Please select at least one scheme to generate a document checklist.")

# Function to initialize the document requirements database
def initialize_document_requirements_db():
    """Initialize the document requirements database with sample data."""
    try:
        # Check if collection exists and has data
        try:
            collection_info = qdrant_client.get_collection("document_requirements")
            if collection_info.vectors_count > 0:
                print("Document requirements collection already exists and has data")
                return True
        except Exception:
            pass
        
        # Create sample dataframe
        df = create_sample_document_requirements()
        
        # Get sample embedding to determine vector size
        sample_text = "Sample document for determining vector size"
        sample_embedding = get_embedding(sample_text)
        
        if sample_embedding is not None:
            vector_size = len(sample_embedding)
            # Create Qdrant collection
            create_document_requirements_collection(vector_size=vector_size)
            
            # Upload data to Qdrant
            upload_document_requirements(df)
            print("Document requirements data initialized successfully!")
            return True
        else:
            print("Failed to generate a sample embedding. Cannot initialize document requirements database.")
            return False
    except Exception as e:
        print(f"Error initializing document requirements database: {e}")
        return False

# Main function to run as standalone script
if __name__ == "__main__":
    # Initialize document requirements database
    initialize_document_requirements_db()
    
    # Set up Streamlit app for testing
    st.set_page_config(page_title="Document Checklist Generator", page_icon="📑", layout="wide")
    add_document_checklist_to_app()