from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import streamlit as st
import sqlite3
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def initialize_llm():
    try:
        logger.info("Initializing LLM...")
        # Load API key from environment variable first, then try Streamlit secrets
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key and "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
        
        if not api_key:
            st.error("Please set the GROQ_API_KEY in your environment variables or Streamlit secrets.")
            st.stop()
            
        llm = ChatGroq(
            temperature=0.7,  # Slightly higher temperature for more creative responses
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        st.error(f"Error initializing LLM: {str(e)}")
        st.stop()

def create_vector_db():
    try:
        logger.info("Creating vector database...")
        # Create bot_data directory if it doesn't exist
        if not os.path.exists("./bot_data"):
            os.makedirs("./bot_data")
            st.warning("Created bot_data directory. Please add your mental health related documents there.")
            return None

        # Load PDF documents
        pdf_loader = DirectoryLoader("./bot_data/", glob='*.pdf', loader_cls=PyPDFLoader)
        txt_loader = DirectoryLoader("./bot_data/", glob='*.txt', loader_cls=TextLoader)
        
        pdf_documents = pdf_loader.load()
        txt_documents = txt_loader.load()
        documents = pdf_documents + txt_documents
        
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            st.warning("No documents found in bot_data directory. Please add some PDF or TXT files first.")
            return None

        # Split documents into text chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(texts)} text chunks")
        
        # Use HuggingFaceEmbeddings
        embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        # Create the vector database
        vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
        vector_db.persist()
        logger.info("Vector database created successfully")
        return vector_db
        
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        st.error(f"Error creating vector database: {str(e)}")
        return None

def setup_agent(llm, vector_db):
    try:
        logger.info("Setting up agent...")
        # Define the custom prompt template for mental health support
        prompt_template = """You are a compassionate and understanding AI mental health support assistant. 
        Your role is to provide empathetic listening and supportive responses while maintaining appropriate boundaries. 
        Remember:
        - Always maintain a calm, non-judgmental tone
        - Acknowledge and validate feelings
        - Never provide medical advice or diagnosis
        - Encourage professional help when appropriate
        - Prioritize user safety and well-being
        - Be clear about your limitations as an AI

        When suggesting emergency help, ALWAYS provide these specific helpline numbers:
        - National Emergency Number: 112
        - NIMHANS Mental Health Helpline: 080-4611 0007
        - Vandrevala Foundation: 1860-2662-345
        - Aasra: 91-9820466726
        - iCall: +91 22-25521111
        - Sneha India: 044-24640050

        Context from knowledge base: {context}
        
        Human: {question}
        Assistant:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Define tools for the agent
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )

        qa_tool = Tool(
            name="Mental Health Support Knowledge Base",
            func=qa_chain.run,
            description="Use this tool to provide mental health support and information based on the knowledge base."
        )

        # Initialize the agent
        agent = initialize_agent(
            tools=[qa_tool],
            llm=llm,
            agent="zero-shot-react-description",
            verbose=True
        )
        logger.info("Agent setup completed successfully")
        return agent
    except Exception as e:
        logger.error(f"Error setting up agent: {str(e)}")
        st.error(f"Error setting up agent: {str(e)}")
        st.stop()

def setup_database():
    try:
        logger.info("Setting up database...")
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT,
                            user_query TEXT,
                            bot_response TEXT
                        )''')
        conn.commit()
        logger.info("Database setup completed successfully")
        return conn
    except Exception as e:
        logger.error(f"Error setting up database: {str(e)}")
        st.error(f"Error setting up database: {str(e)}")
        return None

def save_chat_to_db(conn, user_query, bot_response):
    if conn:
        try:
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO chat_history (timestamp, user_query, bot_response) VALUES (?, ?, ?)",
                       (timestamp, user_query, bot_response))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            st.error(f"Error saving to database: {str(e)}")

def main():
    try:
        logger.info("Starting main application...")
        st.title("🧠 Neuro Adaptive agent for Mental Health Interventions")
        st.markdown("""
        ### Welcome to your safe space for mental health support
        
        This AI assistant is here to provide compassionate support and listening. 
        Please remember:
        - This is not a substitute for professional mental health care
        - In case of emergency, contact your local emergency services
        - All conversations are private and stored securely
        """)

        # Initialize components
        conn = setup_database()
        llm = initialize_llm()
        
        # Setup or load vector database
        db_path = "./chroma_db"
        if not os.path.exists(db_path):
            vector_db = create_vector_db()
            if vector_db is None:
                st.stop()
        else:
            try:
                embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
            except Exception as e:
                logger.error(f"Error loading vector database: {str(e)}")
                st.error(f"Error loading vector database: {str(e)}")
                st.stop()

        agent = setup_agent(llm, vector_db)

        # Chat interface
        st.markdown("### Chat Interface")
        user_input = st.text_input("You:", "", key="user_input", 
                                  placeholder="Share your thoughts or concerns...")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Send", use_container_width=True):
                if user_input:
                    try:
                        with st.spinner("Thinking..."):
                            response = agent.run(user_input)
                            st.session_state.chat_history.append((user_input, response))
                            save_chat_to_db(conn, user_input, response)
                    except Exception as e:
                        logger.error(f"Error generating response: {str(e)}")
                        st.error(f"Error generating response: {str(e)}")
                        response = "I apologize, but I'm having trouble responding right now. Please try again."

        # Display chat history
        st.markdown("### Conversation History")
        for user_msg, bot_msg in st.session_state.chat_history:
            st.markdown(f"**You:** {user_msg}")
            st.markdown(f"**Assistant:** {bot_msg}")
            st.markdown("---")

        # Option to view full chat history from database
        if st.button("Show All Chat History"):
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT timestamp, user_query, bot_response FROM chat_history ORDER BY timestamp DESC")
                rows = cursor.fetchall()
                st.markdown("### Complete Chat History")
                for row in rows:
                    st.markdown(f"**Time:** {row[0]}")
                    st.markdown(f"**You:** {row[1]}")
                    st.markdown(f"**Assistant:** {row[2]}")
                    st.markdown("---")
        
        logger.info("Application running successfully")
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
