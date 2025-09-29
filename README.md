Local rag AI agent for recomending other agents.

# Daily usage
cd "/Users/davidgogi/Desktop/AI agent MVP"
python3 -m venv ai_agent_env
source ai_agent_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2
ollama pull mxbai-embed-large


source ai_agent_env/bin/activate
python main.py
