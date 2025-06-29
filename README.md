# PDF Chat Assistant with RAG

A modern, user-friendly web application that allows you to chat with your PDF documents using Retrieval-Augmented Generation (RAG) technology. Built with Streamlit, LangChain, and Groq.

## Features

- 📄 Upload and process multiple PDF documents
- 💬 Chat with your documents using natural language
- 🔍 Semantic search with document context
- 🧠 Powered by state-of-the-art language models
- 💾 Save and manage chat history
- 🎨 Modern, responsive UI with dark/light mode support

## Prerequisites

- Python 3.8 or higher
- Groq API key (get it from [Groq Console](https://console.groq.com/))
- Optional: HuggingFace API key (for better embedding models)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "4.1-RAG Q&A Conversation"
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here  # Optional
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. In the sidebar:
   - Enter your Groq API key
   - Select a model (default: Gemma2-9b-It)
   - Optionally, set a session ID for different conversations
   - Upload one or more PDF files

4. Start chatting with your documents in the main chat interface

## Features in Detail

### Document Processing
- Supports multiple PDF files
- Automatic text extraction and chunking
- Smart document indexing for efficient retrieval

### Chat Interface
- Clean, modern UI with message bubbles
- Real-time response streaming
- Message history persistence
- Clear chat functionality

### Advanced Features
- Context-aware question answering
- Source document citation
- Session management
- Export chat history as Markdown

## Configuration

### Environment Variables
- `GROQ_API_KEY`: Required for accessing Groq's API
- `HUGGINGFACE_API_KEY`: Optional, for using custom embedding models

### Model Options
- Gemma2-9b-It
- Llama3-8b-8192
- Mixtral-8x7b-32768

## Troubleshooting

### Common Issues
- **API Key Not Working**: Ensure your Groq API key is valid and has sufficient credits
- **Document Not Processing**: Check if the PDF is not password protected and contains extractable text
- **Slow Responses**: Try using a smaller model or reducing the chunk size

### Error Messages
- If you see "NameError: name 'tempfile' is not defined", ensure all required imports are present
- For embedding-related errors, verify your HuggingFace API key if using custom embeddings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Groq](https://groq.com/) and [LangChain](https://www.langchain.com/)

---

Feel free to contribute to this project by submitting issues or pull requests. Happy chatting with your documents! 🚀
