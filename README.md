# Neuro Adaptive Agent for Mental Health Interventions

A compassionate AI-powered mental health support chatbot that provides empathetic listening and supportive responses while maintaining appropriate boundaries.

## Features
- Empathetic conversational AI support
- Knowledge base integration for mental health resources
- Secure chat history storage
- Emergency helpline information
- Vector database for efficient information retrieval

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your GROQ API key to the `.env` file

5. Create a `bot_data` directory and add your mental health related documents (PDF/TXT files)

6. Run the application:
```bash
streamlit run Neuroadaptive_agent/neuro_adaptive.py
```

## Environment Variables
- `GROQ_API_KEY`: Your Groq API key for the LLM service

## Directory Structure
- `Neuroadaptive_agent/`: Main application code
- `bot_data/`: Directory for mental health related documents
- `chroma_db/`: Vector database storage (created automatically)

## Security Notes
- Never commit your `.env` file or any files containing API keys
- The chat history database is stored locally and not included in the repository
- Vector database files are generated locally and not included in the repository
