"""LLM Router for intelligent source selection using DeepSeek"""

from typing import Dict, Any, List
import os
import json
from openai import OpenAI
from rag_system.core.config import get_config

class LLMRouter:
    def __init__(self):
        self.config = get_config()
        api_key = self.config.get('llm.api_key') or os.getenv('DEEPSEEK_API_KEY')
        
        if not api_key:
            raise ValueError("DeepSeek API key not configured. Set DEEPSEEK_API_KEY environment variable.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        self.model = self.config.get('llm.model', 'deepseek-chat')
        self.temperature = self.config.get('llm.temperature', 0.7)
        self.max_tokens = self.config.get('llm.max_tokens', 2000)
    
    def route_query(self, query: str) -> Dict[str, Any]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "select_sources",
                    "description": "Select which data sources to use for answering the query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sources": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["local_knowledge_base", "web_search", "finance", "weather", "transport", "multimodal_ingest"]
                                },
                                "description": "List of sources to query"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation for source selection"
                            }
                        },
                        "required": ["sources", "reasoning"]
                    }
                }
            }
        ]
        
        system_prompt = """You are an intelligent router that selects the best data sources for answering user queries.

Available sources:
- local_knowledge_base: Internal documents and knowledge base
- web_search: Real-time web search for current information
- finance: Stock prices, market data, company financials
- weather: Weather forecasts and historical weather data
- transport: Routes, directions, travel times
- multimodal_ingest: Process uploaded files (PDFs, images, documents)

Selection guidelines:
- Contains stock ticker or financial terms? → finance
- Contains "weather", city name, or date with weather context? → weather
- Contains route, address, or travel query? → transport
- File attached or document processing needed? → multimodal_ingest
- Needs current/recent information? → web_search + local_knowledge_base
- General knowledge query? → local_knowledge_base
- Can select multiple sources if needed

Select the most appropriate sources for the query."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "select_sources"}},
                temperature=self.temperature
            )
            
            tool_call = response.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            
            return {
                'sources': arguments.get('sources', ['local_knowledge_base']),
                'reasoning': arguments.get('reasoning', ''),
                'query': query
            }
        except Exception as e:
            return {
                'sources': ['local_knowledge_base'],
                'reasoning': f'Error in routing: {str(e)}. Defaulting to local knowledge base.',
                'query': query
            }
    
    def synthesize_answer(self, query: str, context: List[Dict[str, Any]], citations: List[str]) -> str:
        context_text = "\n\n".join([
            f"[{i+1}] {doc.get('text', '')}\nSource: {doc.get('source', 'Unknown')}"
            for i, doc in enumerate(context)
        ])
        
        system_prompt = """You are a helpful AI assistant that provides accurate, well-cited answers based on the provided context.

Guidelines:
- Answer based ONLY on the provided context
- Cite sources using [1], [2], etc. for each fact
- If context doesn't contain the answer, say so
- Be concise but comprehensive
- Use clear, professional language"""

        user_prompt = f"""Query: {query}

Context:
{context_text}

Provide a well-cited answer to the query based on the context above."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

_llm_router = None

def get_llm_router() -> LLMRouter:
    global _llm_router
    if _llm_router is None:
        _llm_router = LLMRouter()
    return _llm_router
