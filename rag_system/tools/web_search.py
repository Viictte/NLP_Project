"""Web search tool using Tavily API"""

from typing import Dict, Any, List, Optional
import requests
from rag_system.core.config import get_config
import os

class WebSearchTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.web_search.enabled', True)
        self.api_key = self.config.get('tools.web_search.api_key') or os.getenv('TAVILY_API_KEY')
    
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Web search tool is disabled'}
        
        if not self.api_key:
            return {
                'query': query,
                'results': [],
                'error': 'Tavily API key not configured. Web search unavailable.'
            }
        
        try:
            url = "https://api.tavily.com/search"
            headers = {
                'Content-Type': 'application/json'
            }
            body = {
                'api_key': self.api_key,
                'query': query,
                'max_results': max_results,
                'search_depth': 'basic',
                'include_answer': True
            }
            
            response = requests.post(url, json=body, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'query': query,
                'results': data.get('results', []),
                'answer': data.get('answer', '')
            }
        except Exception as e:
            return {'error': str(e), 'query': query, 'results': []}

_web_search_tool = None

def get_web_search_tool() -> WebSearchTool:
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool
