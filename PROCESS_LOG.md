# Process Log
## AI tools used
ChatGPT — used for implementation planning, debugging LangGraph and MCP integration, troubleshooting deployment issues, designing evaluation scripts, and organizing documentation.
## Development narrative

I developed CineScope as a multi-agent movie research assistant using LangGraph, Chroma, SQLite, Tavily Remote MCP, and a FastAPI web interface. I separated the system into three agents: Research for semantic movie analysis, Discovery for structured database queries, and News for current information. A supervisor selects the relevant agents and combines their results.

I started with a small movie dataset, tested each agent independently, and gradually expanded the corpus to 75 films. I then rebuilt the SQLite database and Chroma index, added model routing between OpenAI and NVIDIA Nemotron, and created a browser interface that streams responses and displays agent execution traces.

During development, I encountered several issues, including Tavily rate limits, incompatible RAGAS metric names, a broken RAG command-line entry point, and Cloud Run authentication settings. I used error messages, logs, and actual API responses to diagnose and fix these problems. I also containerized the application and deployed it to Google Cloud Run.

## Verification

I tested movie discovery filters, semantic research questions about The Matrix, Parasite, and WALL-E, and current movie news searches. A fixed 15-question RAGAS evaluation achieved 0.949 faithfulness and 0.867 context recall. Benchmarks showed an average local RAG response time of 3.918 seconds, while warm Cloud Run requests averaged 21.151 seconds end to end.

## What I learned

I learned that successful tool discovery does not guarantee successful execution, and that evaluation results must be interpreted alongside retrieval limitations and latency measurements. I verified AI-generated suggestions through tests and logs. Before submission, I reviewed verified Discovery Agent error handling, and checked the repository for sensitive or unnecessary files.