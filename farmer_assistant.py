import os
import pandas as pd
import streamlit as st
from googletrans import Translator
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import the document checklist module
from document_checklist import (
    initialize_document_requirements_db,
    generate_document_checklist,
    format_document_checklist,
    add_document_checklist_to_app
)

# At the top of your file, after imports
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the translator
translator = Translator()

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="https://ae245ab4-d048-42a9-bff3-f32bb853dd3f.eu-west-2-0.aws.cloud.qdrant.io", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.6Uho4S7-1Yjne4NcJZdLOioFDy2nkfpkPINi39rjzPw"
)

# Set up Gemini model for chat
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Scheme comparison data - Add comprehensive data for each scheme
SCHEME_DATA = {
    "PM-KISAN": {
        "description": "Direct income support of ₹6,000 per year to farmer families",
        "benefits": "• ₹6,000 per year in three installments\n• Direct transfer to bank accounts\n• No loan component",
        "eligibility": "All landholding farmer families with cultivable land",
        "processing_time": "2-3 months for first payment after registration",
        "application_effort": "Medium - requires land records and bank details",
        "documents_required": "Aadhaar card, land records, bank account details",
        "renewal": "Annual self-declaration required",
        "effort_score": 2,  # 1=Low, 2=Medium, 3=High
        "processing_score": 3,  # 1=Fast, 2=Moderate, 3=Slow
        "renewal_score": 1,  # 1=Easy, 2=Moderate, 3=Difficult
        "benefit_score": 3,  # 1=Low, 2=Medium, 3=High
    },
    "Kisan Credit Card": {
        "description": "Credit card for farmers to access timely credit for agricultural needs",
        "benefits": "• Flexible credit up to ₹3 lakhs\n• Interest subvention of 2%\n• Additional 3% subvention for timely repayment\n• Crop insurance coverage",
        "eligibility": "All farmers, tenant farmers, sharecroppers, and self-help groups",
        "processing_time": "2-4 weeks after application submission",
        "application_effort": "High - requires multiple documents and bank visits",
        "documents_required": "Land records, identity proof, address proof, passport photos, bank account details",
        "renewal": "Every 3-5 years with annual interest payments",
        "effort_score": 3,
        "processing_score": 2,
        "renewal_score": 2,
        "benefit_score": 3,
    },
    "PM Fasal Bima Yojana": {
        "description": "Crop insurance scheme to provide financial support in case of crop failure",
        "benefits": "• Comprehensive risk coverage for pre-sowing to post-harvest losses\n• Low premium rates\n• Full sum insured for total crop loss",
        "eligibility": "All farmers growing notified crops in notified areas",
        "processing_time": "3-6 months for claim settlements after crop cutting experiments",
        "application_effort": "Low - can be bundled with crop loans",
        "documents_required": "Bank account details, land records (for non-loanee farmers)",
        "renewal": "Seasonal enrollment for each crop season",
        "effort_score": 1,
        "processing_score": 3,
        "renewal_score": 2,
        "benefit_score": 2,
    },
    "Soil Health Card Scheme": {
        "description": "Provides soil health cards to farmers with crop-wise recommendations for nutrients",
        "benefits": "• Free soil testing\n• Crop-specific nutrient recommendations\n• Reduces input costs by optimizing fertilizer use",
        "eligibility": "All farmers with agricultural land",
        "processing_time": "1-2 months to receive soil health card",
        "application_effort": "Low - simple registration at local agriculture office",
        "documents_required": "Land location details, identity proof",
        "renewal": "Every 3 years for retesting soil",
        "effort_score": 1,
        "processing_score": 2,
        "renewal_score": 1,
        "benefit_score": 1,
    },
    "Per Drop More Crop": {
        "description": "Micro-irrigation scheme for efficient water use in agriculture",
        "benefits": "• Subsidy of up to 55% for small/marginal farmers\n• 45% subsidy for other farmers\n• Water saving and yield improvement",
        "eligibility": "All farmers with suitable land for micro-irrigation",
        "processing_time": "2-3 months for subsidy approval and installation",
        "application_effort": "High - requires technical evaluation and vendor coordination",
        "documents_required": "Land records, bank details, identity proof, water source proof",
        "renewal": "One-time application, no regular renewal needed",
        "effort_score": 3,
        "processing_score": 2,
        "renewal_score": 1,
        "benefit_score": 2,
    },
    "National Mission for Sustainable Agriculture": {
        "description": "Promotes sustainable farming practices adapted to climate change",
        "benefits": "• Financial assistance for various sustainable practices\n• Training and capacity building\n• Support for organic farming certification",
        "eligibility": "Farmers willing to adopt climate-resilient agricultural practices",
        "processing_time": "1-3 months depending on component",
        "application_effort": "Medium - requires detailed farm plans",
        "documents_required": "Land details, bank account, identity proof, farm plan",
        "renewal": "Component-specific, typically annual for continuing components",
        "effort_score": 2,
        "processing_score": 2,
        "renewal_score": 2,
        "benefit_score": 2,
    }
}

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
            print(f"Type: {type(embedding_result)}")
            print(f"Dir: {dir(embedding_result)}")
            return None
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def create_collection(collection_name="farmer_subsidies", vector_size=768):
    """Create a collection in Qdrant for storing subsidy embeddings."""
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

def process_csv_data(file_path):
    """Process the CSV data file and extract relevant information for the RAG system."""
    df = pd.read_csv(file_path)
    
    # Display the first few rows to understand the data structure
    print("CSV Data Preview:")
    print(df.head())
    
    return df

def upload_data_to_qdrant(df, collection_name="farmer_subsidies"):
    """Upload the data from the dataframe to Qdrant."""
    points = []
    
    # Assuming the dataframe has columns like 'subsidy_name', 'description', 'eligibility', etc.
    for idx, row in df.iterrows():
        # Combine all text fields into a single document
        document = f"Subsidy: {row.get('subsidy_name', '')}\n"
        document += f"Description: {row.get('description', '')}\n"
        document += f"Eligibility: {row.get('eligibility', '')}\n"
        document += f"Benefits: {row.get('benefits', '')}\n"
        document += f"Application Process: {row.get('application_process', '')}\n"
        document += f"Contact: {row.get('contact', '')}\n"
        
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
                    "subsidy_name": row.get("subsidy_name", ""),
                    "description": row.get("description", ""),
                    "eligibility": row.get("eligibility", ""),
                    "benefits": row.get("benefits", ""),
                    "application_process": row.get("application_process", ""),
                    "contact": row.get("contact", ""),
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
    
    print(f"Uploaded {df.shape[0]} records to Qdrant")

def search_qdrant(query, collection_name="farmer_subsidies", limit=5):
    """Search for relevant information in Qdrant based on the query."""
    query_embedding = get_embedding(query)
    
    # Check if embedding was generated successfully
    if query_embedding is None:
        print("Failed to generate embedding for search query")
        return []
    
    try:
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Return directly as search already returns a list of points
        return search_results
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        return []

def detect_language(text):
    """Detect the language of the input text."""
    try:
        # Simple language detection based on common words
        hi_words = ["में", "है", "का", "की", "और", "को", "से", "के", "एक", "पर"]
        mr_words = ["आहे", "मध्ये", "आणि", "एक", "त्या", "तो", "ती", "हे", "या", "ते"]
        
        # Check for Hindi or Marathi words
        for word in hi_words:
            if word in text:
                return "hi"
        
        for word in mr_words:
            if word in text:
                return "mr"
                
        # Default to English
        return "en"
    except Exception as e:
        print(f"Language detection error: {e}")
        return "en"

def translate_text(text, target_language="en"):
    """Translate text to the target language."""
    if not text.strip():
        return ""
    
    if target_language == "en":
        return text
    
    # For this simplified version, we'll skip actual translation
    # and just return the original text with a note
    return f"{text}\n\n[Translation feature is disabled in this version to avoid errors.]"

def generate_answer(query, context, lang="en"):
    """Generate an answer using Gemini API based on the retrieved context."""
    # Handle API quota errors gracefully
    try:
        # Detect language
        detected_lang = detect_language(query)
        
        # For simplicity, we'll use the original query without translation
        query_en = query
        
        # If context is empty, return a default message
        if not context.strip():
            default_response = "I'm sorry, but I couldn't find specific information about that in my database. Please try asking a more general question about agricultural subsidies, or check with your local agricultural office."
            return default_response
        
        # Prepare the prompt with the context and query
        prompt = f"""
        You are an AI assistant specializing in agricultural subsidies for farmers in India.
        
        Context information:
        {context}
        
        User query: {query_en}
        
        Based on the context provided, answer the user's query in a helpful and informative way.
        Focus only on the information provided in the context. If the information isn't available
        in the context, politely state that you don't have that specific information.
        Provide a concise response that directly addresses the query.
        """
        
        try:
            # Generate response with Gemini
            response = gemini_model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 300, "temperature": 0.2}
            )
            answer = response.text
            
            # Note on language
            if detected_lang != "en":
                answer += "\n\n[Note: Translation feature is disabled in this version. Full language support will be available in the next update.]"
            
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            
            # Check if it's a quota error
            if "429" in str(e) and "quota" in str(e).lower():
                return "I'm currently experiencing high demand. The API quota has been exceeded. Please try again in a few minutes or contact the administrator to upgrade the API plan."
            
            return "Sorry, I encountered an error while generating your answer. Please try again."
            
    except Exception as general_e:
        print(f"General error in generate_answer: {general_e}")
        return "An unexpected error occurred. Please try again later."

def get_answer(query, lang="en"):
    """Process a query and generate an answer."""
    # Search for relevant information in Qdrant
    search_results = search_qdrant(query)
    
    # Extract the context from search results
    context = ""
    for result in search_results:
        context += result.payload.get("full_text", "") + "\n\n"
    
    # Generate the answer
    answer = generate_answer(query, context, lang)
    
    return answer

def compare_schemes(schemes_to_compare):
    """Generate comparison data for selected schemes."""
    if len(schemes_to_compare) < 2:
        return None
    
    comparison_data = {scheme: SCHEME_DATA.get(scheme, {}) for scheme in schemes_to_compare}
    return comparison_data

def render_comparison_chart(comparison_data):
    """Create a visual representation of the comparison."""
    if not comparison_data:
        return
    
    # Create two columns for the chart
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create a DataFrame for the radar chart data
        chart_data = []
        for scheme, data in comparison_data.items():
            if not data:  # Skip if scheme data is missing
                continue
                
            chart_data.append({
                "Scheme": scheme,
                "Benefit Level": data.get("benefit_score", 0),
                "Application Effort (Lower is Better)": 4 - data.get("effort_score", 0),  # Invert for better visualization
                "Processing Speed (Lower is Better)": 4 - data.get("processing_score", 0),  # Invert for better visualization
                "Renewal Simplicity (Lower is Better)": 4 - data.get("renewal_score", 0)   # Invert for better visualization
            })
        
        if chart_data:
            # Convert to DataFrame for plotting
            df = pd.DataFrame(chart_data)
            
            # Create a radar chart using Streamlit's native plotting capabilities
            st.subheader("Scheme Comparison Chart")
            st.write("Higher values indicate better performance in each category")
            
            # Use Streamlit's line chart with markers as a substitute for radar chart
            df_plot = df.set_index("Scheme")
            st.line_chart(df_plot)
            
            # Add a note about the interpretation
            st.caption("Note: For Application Effort, Processing Speed, and Renewal Simplicity, the values are inverted so higher is better (easier/faster).")
    
    with col2:
        # Display a legend with score meanings
        st.subheader("Score Legend")
        st.markdown("""
        **Benefit Level**:
        - 1: Low benefits
        - 2: Medium benefits
        - 3: High benefits
        
        **Application Effort**:
        - 3: Easy application
        - 2: Medium effort
        - 1: Difficult application
        
        **Processing Speed**:
        - 3: Fast processing
        - 2: Medium wait time
        - 1: Slow processing
        
        **Renewal Simplicity**:
        - 3: Easy renewal
        - 2: Medium complexity
        - 1: Complex renewal
        """)

def render_scheme_comparison_table(comparison_data):
    """Render a detailed side-by-side comparison table."""
    if not comparison_data:
        return
    
    # Create tabs for different aspects of comparison
    tab1, tab2, tab3, tab4 = st.tabs(["Benefits", "Application Process", "Renewal Requirements", "Full Comparison"])
    
    # Tab 1: Benefits Comparison
    with tab1:
        st.subheader("Benefit Comparison")
        
        # Create columns for each scheme
        cols = st.columns(len(comparison_data))
        
        # Display scheme information in each column
        for i, (scheme, data) in enumerate(comparison_data.items()):
            with cols[i]:
                st.markdown(f"### {scheme}")
                st.markdown(f"**Description**: {data.get('description', 'N/A')}")
                st.markdown("**Benefits**:")
                st.markdown(data.get("benefits", "N/A"))
                
                # Visual indicator for benefit level
                benefit_score = data.get("benefit_score", 0)
                if benefit_score == 3:
                    st.markdown("**Benefit Level**: 🟢🟢🟢 High")
                elif benefit_score == 2:
                    st.markdown("**Benefit Level**: 🟢🟢⚪ Medium")
                else:
                    st.markdown("**Benefit Level**: 🟢⚪⚪ Basic")
    
    # Tab 2: Application Process
    with tab2:
        st.subheader("Application Process Comparison")
        
        # Create columns for each scheme
        cols = st.columns(len(comparison_data))
        
        # Display scheme information in each column
        for i, (scheme, data) in enumerate(comparison_data.items()):
            with cols[i]:
                st.markdown(f"### {scheme}")
                
                # Visual indicators for application effort
                effort_score = data.get("effort_score", 0)
                if effort_score == 1:
                    effort_indicator = "🟢 Low Effort"
                elif effort_score == 2:
                    effort_indicator = "🟡 Medium Effort"
                else:
                    effort_indicator = "🔴 High Effort"
                    
                st.markdown(f"**Application Effort**: {effort_indicator}")
                st.markdown(f"**Processing Time**: {data.get('processing_time', 'N/A')}")
                st.markdown("**Documents Required**:")
                st.markdown(data.get("documents_required", "N/A"))
                
                # Visual indicator for processing time
                proc_score = data.get("processing_score", 0)
                if proc_score == 1:
                    st.markdown("**Processing Speed**: ⚡ Fast")
                elif proc_score == 2:
                    st.markdown("**Processing Speed**: ⏱️ Medium")
                else:
                    st.markdown("**Processing Speed**: ⏳ Slow")
    
    # Tab 3: Renewal Requirements
    with tab3:
        st.subheader("Renewal Requirements Comparison")
        
        # Create columns for each scheme
        cols = st.columns(len(comparison_data))
        
        # Display scheme information in each column
        for i, (scheme, data) in enumerate(comparison_data.items()):
            with cols[i]:
                st.markdown(f"### {scheme}")
                
                st.markdown("**Renewal Process**:")
                st.markdown(data.get("renewal", "N/A"))
                
                # Visual indicator for renewal complexity
                renewal_score = data.get("renewal_score", 0)
                if renewal_score == 1:
                    st.markdown("**Renewal Complexity**: 🟢 Simple")
                elif renewal_score == 2:
                    st.markdown("**Renewal Complexity**: 🟡 Moderate")
                else:
                    st.markdown("**Renewal Complexity**: 🔴 Complex")
    
    # Tab 4: Full Comparison Table
    with tab4:
        st.subheader("Full Comparison Table")
        
        # Prepare data for a comprehensive table
        table_data = []
        for scheme, data in comparison_data.items():
            row = {
                "Scheme": scheme,
                "Description": data.get("description", "N/A"),
                "Benefits": data.get("benefits", "N/A").replace("\n", " "),
                "Application Effort": ["Low", "Medium", "High"][data.get("effort_score", 1) - 1],
                "Processing Time": data.get("processing_time", "N/A"),
                "Documents Required": data.get("documents_required", "N/A"),
                "Renewal Requirements": data.get("renewal", "N/A")
            }
            table_data.append(row)
        
        # Convert to DataFrame and display
        if table_data:
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table)
            
            # Create a download button for the comparison
            csv = df_table.to_csv(index=False)
            st.download_button(
                label="Download Comparison as CSV",
                data=csv,
                file_name="scheme_comparison.csv",
                mime="text/csv"
            )

def build_streamlit_app():
    """Build the Streamlit app interface."""
    st.set_page_config(page_title="Farmer Subsidy Assistant", page_icon="🌾", layout="wide")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["💬 Ask about Subsidies", "📑 Document Checklist", "🔄 Compare Schemes"])
    
    # Tab 1: Chatbot Interface
    with tab1:
        st.title("🌾 Farmer Subsidy Assistant")
        st.subheader("Ask questions about agricultural subsidies in English, Hindi, or Marathi")
        
        # Sidebar for settings
        st.sidebar.title("Chatbot Settings")
        language = st.sidebar.selectbox(
            "Preferred Language",
            ["English", "Hindi", "Marathi"],
            index=0,
            key="chat_language"
        )
        
        # Map language selection to language codes
        lang_map = {
            "English": "en",
            "Hindi": "hi",
            "Marathi": "mr"
        }
        selected_lang = lang_map[language]
        
        # Notice about translation feature
        if selected_lang != "en":
            st.sidebar.warning("Note: Translation features are limited in this version to avoid errors. The system will understand basic words in your language but respond primarily in English.")
        
        # Display information about vector database
        with st.sidebar.expander("Database Information"):
            try:
                collection_info = qdrant_client.get_collection("farmer_subsidies")
                st.write(f"Collection: farmer_subsidies")
                st.write(f"Vectors: {collection_info.vectors_count}")
                st.write(f"Status: Connected ✅")
            except Exception:
                st.write("Collection: Not found")
                st.write("Status: Not connected ❌")
        
        # File uploader for CSV data
        with st.sidebar.expander("Upload Subsidy Data"):
            uploaded_file = st.file_uploader("Upload CSV file with subsidy data", type=["csv"], key="subsidy_csv")
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                with open("temp_data.csv", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                if st.button("Process Subsidy Data"):
                    with st.spinner("Processing data and creating embeddings..."):
                        # Process the CSV data
                        df = process_csv_data("temp_data.csv")
                        
                        # Get sample embedding to determine vector size
                        sample_text = "Sample document for determining vector size"
                        sample_embedding = get_embedding(sample_text)
                        
                        if sample_embedding is not None:
                            vector_size = len(sample_embedding)
                            # Create Qdrant collection
                            create_collection(vector_size=vector_size)
                            
                            # Upload data to Qdrant
                            upload_data_to_qdrant(df)
                            st.success("Data processed and uploaded to Qdrant successfully!")
                        else:
                            st.error("Failed to generate a sample embedding. Please check your API key and internet connection.")
        
        # Initialize or get chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Process any new input
        if prompt := st.chat_input("Type your question here..."):
            # Append user message to state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message (immediate feedback)
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_answer(prompt, selected_lang)
                    st.markdown(response)
            
            # Append assistant message to state
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force a rerun to ensure the chat input appears at the bottom
            st.rerun()
    
    # Tab 2: Document Checklist Generator
    with tab2:
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
            default=["PM-KISAN"],
            key="checklist_schemes"
        )
        
        # Add a button to generate the checklist
        if st.button("Generate Document Checklist"):
            if selected_schemes:
                with st.spinner("Generating your personalized document checklist..."):
                    # Check if document requirements collection exists
                    try:
                        qdrant_client.get_collection("document_requirements")
                    except Exception:
                        # Initialize document requirements database if it doesn't exist
                        st.info("Initializing document requirements database...")
                        initialize_document_requirements_db()
                    
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
        
        # Add a helpful explanation
        with st.expander("How to use the Document Checklist Generator"):
            st.markdown("""
            ### How to Use This Tool
            
            1. Select one or more subsidy schemes you're interested in applying for
            2. Click the "Generate Document Checklist" button
            3. Review the personalized checklist that shows all required documents
            4. Pay attention to difficulty levels and processing times
            5. Download the checklist to keep it for reference
            
            ### Understanding the Checklist
            
            - 🟢 **Easy:** Documents that are simple to obtain
            - 🟡 **Medium:** Documents that require some effort
            - 🔴 **Hard:** Documents that are more challenging to obtain
            
            - ⚡ **Quick:** Processing time of 1-3 days
            - ⏱️ **Moderate:** Processing time of 1-2 weeks
            - ⏳ **Long:** Processing time of more than a month
            
            - 🆓 **Free:** No cost to obtain
            - 💰 **Paid:** Documents that have a cost associated
            """)
    
    # Tab 3: Scheme Comparison Tool
    with tab3:
        st.title("🔄 Scheme Comparison Tool")
        st.subheader("Compare different agricultural subsidy schemes side by side")
        
        # Get available schemes
        available_schemes = list(SCHEME_DATA.keys())
        
        # Create multi-select for schemes to compare
        schemes_to_compare = st.multiselect(
            "Select 2-3 schemes to compare:",
            available_schemes,
            default=available_schemes[:2] if len(available_schemes) >= 2 else available_schemes,
            max_selections=3,
            key="comparison_schemes"
        )
        
        # Validate selection and show comparison
        if len(schemes_to_compare) < 2:
            st.warning("Please select at least 2 schemes to compare.")
        else:
            # Get comparison data
            comparison_data = compare_schemes(schemes_to_compare)
            
            # Display visual comparison chart
            render_comparison_chart(comparison_data)
            
            # Display detailed comparison table
            render_scheme_comparison_table(comparison_data)
            
            # Add an explanation section
            with st.expander("Understanding the Comparison"):
                st.markdown("""
                ### How to Use This Comparison Tool
                
                This tool provides a side-by-side comparison of different agricultural subsidy schemes to help you make informed decisions:
                
                - **Benefits Tab**: Compare what each scheme offers and the level of benefits
                - **Application Process Tab**: Compare how difficult it is to apply and how long it takes to process
                - **Renewal Requirements Tab**: Compare how often and how complex the renewal process is
                - **Full Comparison Tab**: View all details in a single table format that you can download
                
                ### Indicators Explained
                
                **Benefit Level**:
                - 🟢🟢🟢 High - Substantial financial or material benefits
                - 🟢🟢⚪ Medium - Moderate benefits
                - 🟢⚪⚪ Basic - Limited but still valuable benefits
                
                **Application Effort**:
                - 🟢 Low - Few documents, simple process
                - 🟡 Medium - Several documents, moderate complexity
                - 🔴 High - Many documents, complex process
                
                **Processing Speed**:
                - ⚡ Fast - Quick approval and implementation
                - ⏱️ Medium - Standard processing time
                - ⏳ Slow - Long waiting periods
                
                **Renewal Complexity**:
                - 🟢 Simple - Easy renewal process
                - 🟡 Moderate - Some effort required 
                - 🔴 Complex - Difficult renewal process
                """)
            
            # Add comparison-based recommendations
            st.subheader("Recommendations Based on Comparison")
            
            # Generate personalized recommendations based on the comparison
            recommendations = {
                "Easy Access": min(comparison_data.items(), key=lambda x: x[1].get("effort_score", 3))[0],
                "Fast Processing": min(comparison_data.items(), key=lambda x: x[1].get("processing_score", 3))[0],
                "High Benefits": max(comparison_data.items(), key=lambda x: x[1].get("benefit_score", 0))[0],
                "Simple Renewal": min(comparison_data.items(), key=lambda x: x[1].get("renewal_score", 3))[0]
            }
            
            # Display recommendations
            cols = st.columns(4)
            with cols[0]:
                st.markdown(f"**Easiest Application**<br>{recommendations['Easy Access']}", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"**Fastest Processing**<br>{recommendations['Fast Processing']}", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"**Highest Benefits**<br>{recommendations['High Benefits']}", unsafe_allow_html=True)
            with cols[3]:
                st.markdown(f"**Simplest Renewal**<br>{recommendations['Simple Renewal']}", unsafe_allow_html=True)

# Main function to run the application
if __name__ == "__main__":
    build_streamlit_app()