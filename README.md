# 🛒 ProductPulse – A Unified E-Commerce Intelligence System

## 👥 Team 4
- **Yash Khavnekar** – Data Collection, Web Scraping, Sentiment Analysis (MCP)
- **Shushil Girish** – Agent Integration, Backend + ETL (LangGraph, FastAPI, Airflow)
- **Riya Mate** – Frontend, Documentation, Codelabs

---

## 📌 Project Overview

**ProductPulse** is an AI-powered, dual-user e-commerce intelligence platform that merges structured insights from **Snowflake Marketplace** with unstructured web data like user reviews and real-time price comparisons. Built using modern data and AI tooling, it serves both **customers** and **vendors** through customized, responsive interfaces.

---

## 🧠 Key Features

### For Customers
- 🔍 Search products and compare real-time pricing across platforms (Amazon, Walmart, etc.)
- 💬 Summarized user reviews with pros, cons, and sentiment scores
- 📊 Dynamic pricing reports and visualizations

### For Vendors
- 📈 KPIs: CTR, conversion rate, sales rank, price index
- 🧠 AI-generated insights based on historical and competitive data
- 📦 Sentiment trajectory, complaint tracking, and improvement suggestions

---

## 🔧 Architecture Overview

![alt text](image.png)

- **Frontend**: Streamlit (Customers), React + Tailwind (Vendors)
- **Backend**: FastAPI
- **Agents**: LangGraph + Model Context Protocol (MCP)
- **ETL Pipelines**: Apache Airflow
- **Data Sources**:
  - Structured: [Similarweb Amazon Dataset (Snowflake Marketplace)](https://app.snowflake.com/marketplace/listing/GZT1ZA3NK6/similarweb-ltd-amazon-and-e-commerce-websites-product-views-and-purchases)
  - Unstructured: Web scraping (user reviews), Web search (pricing)

---

## 🧱 Tech Stack

| Layer              | Tools                                                                 |
|-------------------|------------------------------------------------------------------------|
| Backend API       | FastAPI, LangGraph, HuggingFace Transformers, MCP                     |
| Frontend          | Streamlit, React, TailwindCSS                                          |
| Agents/LLMs       | GPT-4 via OpenAI, LangChain Tool Abstraction                           |
| ETL & Scheduling  | Apache Airflow (GCP Composer)                                          |
| Storage           | Snowflake, PostgreSQL, AWS S3, Redis, Pinecone                        |
| CI/CD & DevOps    | Docker, GitHub Actions, GCP Cloud Run, Artifact Registry              |

---

## 🧩 System Flow

1. User enters product query via Streamlit or vendor dashboard.
2. FastAPI triggers LangGraph agents based on user role.
3. Agents fetch:
   - Structured KPIs from Snowflake
   - Reviews via web scraping
   - Pricing via web search
4. Data processed and returned via summarization pipelines.
5. Dashboard updates with real-time analytics, rankings, and insights.

---

## 🚀 Deployment

- GCP Cloud Run: FastAPI + Streamlit containers
- Cloud Composer: DAG orchestration via Apache Airflow
- GitHub Actions: CI/CD for code and pipeline updates
- Secrets & Caching: GCP Secret Manager + Redis

---

## 🗓️ Project Timeline

| Date       | Milestone                                      |
|------------|------------------------------------------------|
| Apr 1–4    | Setup, Dataset access, GitHub repo creation    |
| Apr 5–7    | ETL pipeline (Airflow + Snowflake)             |
| Apr 8–9    | Backend API + KPI analysis                     |
| Apr 10     | Sentiment analysis pipeline                    |
| Apr 11     | Streamlit UI + integration                     |
| Apr 12–13  | CI/CD setup, cloud deployment                  |
| Apr 14     | Final testing, documentation, and demo video   |

---

## ✅ Goals

- Real-time e-commerce dashboard for product analysis
- Dual-agent system tailored for buyers and vendors
- 85%+ sentiment accuracy on labeled data
- Automated, scalable data pipelines with minimal latency

---
## 📚 Detailed Proeject Explanation

- [Google CodeLabs](https://codelabs-preview.appspot.com/?file_id=1_936snjPYvoj-RmfO5Vcm2G8xzjVTv0XGRy5wHlFiCo#0)

---

## 📚 Resources & References

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Airflow + Snowflake for E-Commerce](https://www.astronomer.io/docs/learn/reference-architecture-elt-snowflake/)
- [LangGraph + MCP](https://changelog.langchain.com/announcements/mcp-adapters-for-langchain-and-langgraph)
- [Deploy FastAPI to Cloud Run](https://testdriven.io/blog/deploying-fastapi-to-cloud-run/)

---

## ⚠️ Known Challenges & Solutions

| Challenge | Mitigation |
|----------|------------|
| Web scraping block | User-agents, proxies, rate-limiting |
| Noisy reviews | NLP + LLM summarization |
| Agent reliability | LangGraph workflows + fallback tools |
| LLM rate limits | Async processing + Redis caching |

---

## 📽️ Demo

Coming soon – Stay tuned for a 10-minute walkthrough video!

---

## 📄 License

This project is developed for academic purposes and is shared under the [MIT License](LICENSE).

