from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert AI agent consultant who helps users find the perfect AI agents for their specific needs.

Here are some relevant AI agents that match the user's request: {agents}

User's request: {question}

Based on the user's request and the relevant AI agents provided above, please:

1. Recommend the most suitable AI agents (2-3 agents maximum)
2. For each recommended agent, provide:
   - Agent name
   - Category
   - Brief explanation of why it matches their needs
   - Key features that are relevant to their request
   - Links to the agent's website
3. If no agents are suitable, explain why and suggest what type of agent they might need instead

Be specific and helpful in your recommendations.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print("🤖 AI Agent Recommendation System")
print("=" * 50)
print("I'll help you find the perfect AI agents for your needs!")
print("Type 'q' to quit\n")

while True:
    print("\n" + "="*50)
    question = input("What kind of AI agent do you need help with? ")
    
    if question.lower() == 'q':
        print("Thanks for using the AI Agent Recommendation System! 👋")
        break
    
    print("\n🔍 Searching for relevant AI agents...")
    
    # Retrieve relevant agents
    agents = retriever.invoke(question)
    
    # Format the agents for the prompt
    agents_text = ""
    for i, agent in enumerate(agents, 1):
        agents_text += f"\n{i}. {agent.metadata['name']}\n"
        agents_text += f"   Category: {agent.metadata['category']}\n"
        agents_text += f"   Description: {agent.metadata['description']}\n"
        agents_text += f"   Links: {agent.metadata['links']}\n"
        agents_text += "-" * 40 + "\n"
    
    print("💡 Generating recommendations...")
    
    # Get recommendations
    result = chain.invoke({"agents": agents_text, "question": question})
    print(f"\n🎯 AI Agent Recommendations:\n{result}")
