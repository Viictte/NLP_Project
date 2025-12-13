#!/usr/bin/env python3
"""Test script for V2 workflow with new test questions"""

import sys
import json
from rag_system.workflows.rag_workflow import get_rag_workflow

# Test questions from WORKFLOW_V2_DESIGN.md
TEST_QUESTIONS = [
    {
        "id": 1,
        "query": "請告訴我最近一屆香港電影金像獎的最佳男主角是誰？並給出他所主演得獎電影在豆瓣上的評分與評價摘要。",
        "expected": "Winner name + film + Douban rating + review summary",
        "category": "Multi-part cross-domain"
    },
    {
        "id": 2,
        "query": "幫我規劃明天在香港中環的一日行程：包含天氣預報、適合的穿著建議，以及從K11 MUSEA去香港科技大學往返的交通路線與時間預估。",
        "expected": "Weather + clothing advice + transport routes with times",
        "category": "Multi-part cross-domain"
    },
    {
        "id": 3,
        "query": "1024 減去 768 等於多少？",
        "expected": "Direct answer: 256, no retrieval",
        "category": "Simple / Direct LLM"
    },
    {
        "id": 4,
        "query": "寫一句鼓勵正在準備考試的學生的話。",
        "expected": "Direct LLM generation, no retrieval",
        "category": "Simple / Direct LLM"
    },
    {
        "id": 5,
        "query": ".",
        "expected": "Direct LLM handles gracefully",
        "category": "Noise input"
    },
    {
        "id": 6,
        "query": "？？？？",
        "expected": "Direct LLM handles gracefully",
        "category": "Noise input"
    },
    {
        "id": 7,
        "query": "請告訴我明天早上香港將軍澳的天氣，並說明是否適合帶小朋友去海邊玩。",
        "expected": "Weather data + recommendation",
        "category": "Single-domain retrieval"
    },
    {
        "id": 8,
        "query": "請查一下騰訊控股（0700.HK）今天的股價大約是多少，以及和上週同一時間相比漲跌幅如何？",
        "expected": "Current price + comparison",
        "category": "Single-domain retrieval"
    },
    {
        "id": 9,
        "query": "我現在在深圳灣口岸，想在三小時內到達香港科技大學，請給我兩條不同的路線選擇（包括時間和交通工具）。",
        "expected": "2 route options with times and modes",
        "category": "Single-domain retrieval"
    },
    {
        "id": 10,
        "query": "今年誰獲得了諾貝爾物理學獎？簡要說明得獎原因。",
        "expected": "Winner + reason",
        "category": "Single-domain retrieval"
    },
    {
        "id": 11,
        "query": "請告訴我最近三屆香港電影金像獎最佳男主角得獎者及其得獎電影，並比較三部電影在豆瓣上的評分高低。",
        "expected": "3 winners + 3 films + 3 ratings + comparison",
        "category": "Evaluator stress test"
    }
]

def test_query(workflow, test_case):
    """Test a single query and return results"""
    print(f"\n{'='*80}")
    print(f"Test {test_case['id']}: {test_case['category']}")
    print(f"Query: {test_case['query']}")
    print(f"Expected: {test_case['expected']}")
    print(f"{'='*80}")
    
    try:
        result = workflow.execute(
            query=test_case['query'],
            strict_local=False,
            fast_mode=False,
            allow_web_search=True
        )
        
        print(f"\n✓ Query executed successfully")
        print(f"Mode: {result.get('query_plan', {}).get('mode', 'N/A')}")
        print(f"Sources used: {result.get('sources_used', [])}")
        print(f"Iterations: {result.get('iterations', 1)}")
        
        if 'evaluation' in result:
            eval_data = result['evaluation']
            print(f"Evaluation - Complete: {eval_data.get('complete', 'N/A')}")
            print(f"Evaluation - Score: {eval_data.get('completeness_score', 'N/A')}")
            if eval_data.get('issues'):
                print(f"Evaluation - Issues: {eval_data['issues']}")
        
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nLatency: {result.get('latency_ms', 0):.0f}ms")
        
        return {
            'success': True,
            'result': result
        }
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def main():
    print("="*80)
    print("V2 Workflow Testing - New Test Questions")
    print("="*80)
    
    workflow = get_rag_workflow()
    
    results = []
    for test_case in TEST_QUESTIONS:
        result = test_query(workflow, test_case)
        results.append({
            'test_id': test_case['id'],
            'category': test_case['category'],
            'query': test_case['query'],
            'success': result['success']
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    total = len(results)
    passed = sum(1 for r in results if r['success'])
    failed = total - passed
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed tests:")
        for r in results:
            if not r['success']:
                print(f"  - Test {r['test_id']}: {r['category']}")
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
