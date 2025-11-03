"""RAG Workflow orchestration using LangGraph"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from rag_system.core.config import get_config
from rag_system.workflows.llm_router import get_llm_router
from rag_system.services.hybrid_retrieval import get_hybrid_retrieval_service
from rag_system.tools.weather import get_weather_tool
from rag_system.tools.finance import get_finance_tool
from rag_system.tools.transport import get_transport_tool
from rag_system.tools.web_search import get_web_search_tool
from rag_system.parsers.document_parser import get_document_parser

class RAGWorkflow:
    def __init__(self):
        self.config = get_config()
        self.llm_router = get_llm_router()
        self.retrieval = get_hybrid_retrieval_service()
        self.weather_tool = get_weather_tool()
        self.finance_tool = get_finance_tool()
        self.transport_tool = get_transport_tool()
        self.web_search_tool = get_web_search_tool()
        self.document_parser = get_document_parser()
    
    def execute(self, query: str, strict_local: bool = False, fast_mode: bool = False) -> Dict[str, Any]:
        start_time = datetime.now()
        
        if strict_local:
            routing = {
                'sources': ['local_knowledge_base'],
                'reasoning': 'Strict local mode enabled',
                'query': query
            }
        else:
            routing = self.llm_router.route_query(query)
        
        sources = routing['sources']
        
        all_context = []
        tool_results = {}
        
        if 'local_knowledge_base' in sources:
            local_docs = self.retrieval.retrieve(query)
            all_context.extend(local_docs)
            tool_results['local_knowledge_base'] = {
                'count': len(local_docs),
                'docs': local_docs
            }
        
        if not strict_local:
            if 'web_search' in sources and not fast_mode:
                web_results = self.web_search_tool.search(query)
                if 'results' in web_results:
                    for result in web_results['results']:
                        all_context.append({
                            'text': result.get('content', result.get('snippet', '')),
                            'source': 'web_search',
                            'url': result.get('url', ''),
                            'credibility_score': 0.6,
                            'final_score': 0.7
                        })
                tool_results['web_search'] = web_results
            
            if 'finance' in sources:
                finance_results = self._handle_finance(query)
                tool_results['finance'] = finance_results
                if 'data' in finance_results:
                    all_context.append({
                        'text': str(finance_results),
                        'source': 'finance',
                        'credibility_score': 0.9,
                        'final_score': 0.85
                    })
            
            if 'weather' in sources:
                weather_results = self._handle_weather(query)
                tool_results['weather'] = weather_results
                if 'data' in weather_results:
                    all_context.append({
                        'text': str(weather_results),
                        'source': 'weather',
                        'credibility_score': 0.85,
                        'final_score': 0.8
                    })
            
            if 'transport' in sources:
                transport_results = self._handle_transport(query)
                tool_results['transport'] = transport_results
                if 'data' in transport_results:
                    all_context.append({
                        'text': str(transport_results),
                        'source': 'transport',
                        'credibility_score': 0.8,
                        'final_score': 0.75
                    })
        
        citations = self._build_citations(all_context)
        
        answer = self.llm_router.synthesize_answer(query, all_context[:10], citations)
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            'query': query,
            'answer': answer,
            'routing': routing,
            'sources_used': sources,
            'tool_results': tool_results,
            'context_count': len(all_context),
            'citations': citations,
            'latency_ms': latency_ms,
            'timestamp': end_time.isoformat()
        }
    
    def _handle_finance(self, query: str) -> Dict[str, Any]:
        tickers = self._extract_tickers(query)
        
        if not tickers:
            return {'error': 'No stock tickers found in query'}
        
        if len(tickers) == 1:
            return self.finance_tool.get_stock_price(tickers[0])
        else:
            return self.finance_tool.compare_stocks(tickers)
    
    def _handle_weather(self, query: str) -> Dict[str, Any]:
        location = self._extract_location(query)
        date = self._extract_date(query)
        
        if not location:
            location = "New York"
        
        return self.weather_tool.get_weather(location, date)
    
    def _handle_transport(self, query: str) -> Dict[str, Any]:
        locations = self._extract_locations(query)
        
        if len(locations) < 2:
            return {'error': 'Need origin and destination for transport query'}
        
        return self.transport_tool.get_route(locations[0], locations[1])
    
    def _extract_tickers(self, query: str) -> List[str]:
        words = query.upper().split()
        common_tickers = ['NVDA', 'AMD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
        
        found_tickers = []
        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word in common_tickers:
                found_tickers.append(clean_word)
        
        return found_tickers
    
    def _extract_location(self, query: str) -> Optional[str]:
        words = query.split()
        
        for i, word in enumerate(words):
            if word.lower() in ['in', 'at', 'for']:
                if i + 1 < len(words):
                    return ' '.join(words[i+1:i+3])
        
        return None
    
    def _extract_locations(self, query: str) -> List[str]:
        locations = []
        
        if ' to ' in query.lower():
            parts = query.lower().split(' to ')
            if len(parts) >= 2:
                locations.append(parts[0].split()[-1])
                locations.append(parts[1].split()[0])
        
        return locations
    
    def _extract_date(self, query: str) -> Optional[str]:
        return None
    
    def _build_citations(self, context: List[Dict[str, Any]]) -> List[str]:
        citations = []
        for i, doc in enumerate(context[:10]):
            source = doc.get('source', 'Unknown')
            url = doc.get('url', '')
            
            if url:
                citations.append(f"[{i+1}] {source}: {url}")
            else:
                citations.append(f"[{i+1}] {source}")
        
        return citations

_rag_workflow = None

def get_rag_workflow() -> RAGWorkflow:
    global _rag_workflow
    if _rag_workflow is None:
        _rag_workflow = RAGWorkflow()
    return _rag_workflow
