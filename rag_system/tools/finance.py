"""Finance tool using yfinance"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
from rag_system.core.config import get_config

class FinanceTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.finance.enabled', True)
    
    def get_stock_price(self, ticker: str, period: str = '1mo') -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Finance tool is disabled'}
        
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return {'error': f'No data found for ticker: {ticker}'}
            
            return {
                'ticker': ticker,
                'period': period,
                'data': hist.to_dict('records'),
                'latest_price': float(hist['Close'].iloc[-1]),
                'change': float(hist['Close'].iloc[-1] - hist['Close'].iloc[0]),
                'change_percent': float((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Finance tool is disabled'}
        
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def compare_stocks(self, tickers: List[str], period: str = '1mo') -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Finance tool is disabled'}
        
        try:
            results = {}
            for ticker in tickers:
                results[ticker] = self.get_stock_price(ticker, period)
            
            return {
                'comparison': results,
                'period': period
            }
        except Exception as e:
            return {'error': str(e)}

_finance_tool = None

def get_finance_tool() -> FinanceTool:
    global _finance_tool
    if _finance_tool is None:
        _finance_tool = FinanceTool()
    return _finance_tool
