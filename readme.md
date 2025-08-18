# Krishi Financial Advisor API 🚜

A smart financial advisor for farmers in India, powered by Google's Gemini model and a RAG (Retrieval-Augmented Generation) pipeline. This API provides tailored financial advice on agricultural loans and government schemes based on a farmer's specific goals and location.

!

[Image of a farmer using a tablet in a field]


---

## ✨ Features

* **Smart Financial Advice**: Leverages Google's Gemini-1.5-Flash model to generate clear, concise, and relevant financial proposals.
* **Location-Aware**: Uses reverse geocoding to identify the farmer's state, enabling state-specific scheme recommendations.
* **Extensible Knowledge Base**: Built on a Chroma vector database, allowing you to easily add more documents about new schemes and policies.
* **Ready for Deployment**: Includes a `Dockerfile` for easy containerization and deployment on cloud platforms like Google Cloud Run.
* **Interactive API**: Built with FastAPI, providing automatic, interactive API documentation (via Swagger UI).

---

## 🛠️ Tech Stack

* **Backend**: Python, FastAPI
* **AI/ML**: Google Generative AI (Gemini), LangChain, Sentence Transformers
* **Database**: ChromaDB (Vector Store)
* **Geolocation**: Geopy
* **Deployment**: Docker

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.9+
* Docker (for containerization)
* A Google API Key with the "Generative Language API" enabled.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd financial_advisor