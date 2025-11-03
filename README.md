# RAG System - Advanced Retrieval Augmented Generation with LLM

A production-ready RAG (Retrieval Augmented Generation) system with hybrid retrieval, intelligent routing, multimodal support, and domain-specific tools.

## Features

- **Hybrid Retrieval**: Combines dense (Qdrant) and sparse (Elasticsearch BM25) retrieval with Reciprocal Rank Fusion (RRF)
- **Advanced Reranking**: Cross-encoder reranking with freshness and credibility scoring
- **Intelligent Routing**: LLM-based source selection using DeepSeek
- **Domain-Specific Tools**: Weather, finance, transport, and web search integrations
- **Multimodal Support**: PDF, HTML, Office documents, images (OCR)
- **Caching**: Redis-based caching for queries, reranking, and answers
- **CLI Interface**: Simple command-line interface for all operations

## Architecture

### Core Services
- **Vector DB**: Qdrant (HNSW + quantization)
- **Keyword Search**: Elasticsearch (BM25)
- **Cache**: Redis
- **LLM**: DeepSeek (routing, tool calls, synthesis)
- **Embeddings**: bge-m3 (multilingual, hybrid-friendly)
- **Reranker**: BAAI/bge-reranker-large (swappable)

### Workflow
1. **Router** → Intelligent source selection
2. **Fetch** → Parallel tool execution (local RAG, web search, finance, weather, transport)
3. **Normalize** → Parse and tabularize results
4. **Rerank & Filter** → Cross-encoder + freshness + credibility scoring
5. **Synthesis** → LLM generates answer with citations
6. **Citations** → Inline citations for each fact

## Installation

### Prerequisites
- Python 3.10+ (3.12 recommended)
- Docker Desktop (running)
- DeepSeek API key (get from https://platform.deepseek.com/)
- Optional: Tesseract OCR for image processing (`brew install tesseract` on macOS)

### Quick Start (Automated Setup)

**Option 1: Using setup script (recommended)**
```bash
git clone https://github.com/Viictte/RAG_LLM.git
cd RAG_LLM
./setup.sh
```

The setup script will:
- Create a virtual environment
- Install all dependencies
- Start Docker services
- Verify system status

**Option 2: Using Makefile**
```bash
git clone https://github.com/Viictte/RAG_LLM.git
cd RAG_LLM
make setup          # Create venv and install dependencies
make up             # Start Docker services
source .venv/bin/activate
make status         # Verify system
```

### Manual Setup

1. Clone the repository:
```bash
git clone https://github.com/Viictte/RAG_LLM.git
cd RAG_LLM
```

2. Create virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY
```

4. Start infrastructure services:
```bash
docker compose up -d
```

5. Verify system status:
```bash
python3 ./rag status
```

**Note for macOS users:**
- Ensure Docker Desktop is running before starting services
- If you need OCR support: `brew install tesseract`
- On Apple Silicon (M1/M2/M3), PyTorch will use CPU mode by default

## Usage

### Ingest Documents

Ingest a single file:
```bash
./rag ingest ./data/corpus/document.pdf
```

Ingest a directory:
```bash
./rag ingest ./data/corpus
```

Ingest a URL:
```bash
./rag ingest https://example.com/article
```

### Ask Questions

Basic query:
```bash
./rag ask "What is the impact of NVIDIA earnings on stock price?"
```

Strict local mode (no web search):
```bash
./rag ask --strict-local "Summarize the internal design memo"
```

Fast mode (skip web search):
```bash
./rag ask --fast "Give me a quick answer"
```

JSON output:
```bash
./rag ask --json "What is the weather in Tokyo?"
```

### Configuration

View configuration:
```bash
./rag config show
```

Get a specific value:
```bash
./rag config get reranker.model
```

Set a value:
```bash
./rag config set reranker.model cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Configuration

Edit `config/config.yaml` to customize:

- LLM settings (model, temperature, max_tokens)
- Embeddings model
- Reranker model and top_k
- Retrieval parameters (top_k, RRF k)
- Scoring weights (cross-encoder, freshness, credibility)
- Chunking parameters
- Domain credibility priors
- Tool enablement

## Domain-Specific Tools

### Weather
Uses Open-Meteo API (no key required):
```bash
./rag ask "What's the weather in Taipei on 2025-11-05?"
```

### Finance
Uses yfinance for stock data:
```bash
./rag ask "Compare NVDA and AMD stock performance"
```

### Transport
Uses OpenRouteService (API key optional):
```bash
./rag ask "Route from San Francisco to Los Angeles"
```

### Web Search
Uses Tavily API (requires key):
```bash
./rag ask "Latest news about AI developments"
```

## Performance

### Speed Optimizations
- HNSW indexing with quantization
- Parallel tool execution
- Redis caching (queries, reranking, answers)
- Batch reranking
- Configurable top_k cutoffs

### Accuracy Optimizations
- RRF hybrid retrieval
- Cross-encoder reranking
- Freshness scoring (exponential decay)
- Credibility scoring (domain priors)
- Deduplication (cosine similarity)
- Mandatory citations

## Development

### Project Structure
```
RAG_LLM/
├── config/              # Configuration files
├── data/                # Data directories
│   ├── corpus/          # Documents to ingest
│   └── uploads/         # Uploaded files
├── rag_system/          # Main package
│   ├── core/            # Configuration management
│   ├── services/        # Vector DB, search, caching
│   ├── parsers/         # Document parsers
│   ├── tools/           # Domain-specific tools
│   ├── workflows/       # LangGraph orchestration
│   └── cli/             # CLI interface
├── tests/               # Test files
├── docker-compose.yml   # Infrastructure services
├── requirements.txt     # Python dependencies
└── rag                  # CLI entry point
```

### Testing

Start infrastructure:
```bash
docker compose up -d
```

Run tests:
```bash
python -m pytest tests/
```

## API Keys

**Required:**
- `DEEPSEEK_API_KEY`: For LLM routing and synthesis (get your key from https://platform.deepseek.com/)

**Optional:**
- `TAVILY_API_KEY`: For web search
- `OPENROUTESERVICE_API_KEY`: For transport routing
- `ALPHA_VANTAGE_API_KEY`: For advanced finance data

**Note:** Without a valid DeepSeek API key, the system will still perform document ingestion, retrieval, and reranking, but answer synthesis will fail. The retrieval system (hybrid search, reranking, caching) works independently and can be tested without an API key.

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or PR.

## Support

For issues or questions, please open a GitHub issue.
