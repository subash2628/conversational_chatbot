# Chatbot Setup Instructions

Follow these steps to set up and run the chatbot:

## 1. Environment Setup

Create a `.env` file in the project root and add the following API keys:

```
GOOGLE_API_KEY="your_google_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
```

## 2. Create Python Virtual Environment

```bash
python3 -m venv chatbot
source chatbot/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Application

Choose one of the following commands based on your preferred model:

### For Gemini model:
```bash
streamlit run just_replica_fine_gemini.py
```

### For OpenAI model:
```bash
streamlit run just_replica_fine_openai.py
```

Enjoy using your chatbot!
