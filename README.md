# 🌐 Nexus RAG — Multilingual Document Intelligence

> A production-grade Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask questions in **Persian**, **English**, or **Arabic** — powered by Groq, LangChain, and ChromaDB.

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 🌐 **Live Demo** | [nexus-rag.streamlit.app](https://nexus-rag.streamlit.app) |
| 📂 **GitHub Repository** | [github.com/A-srt343/persian-rag-chatbot](https://github.com/A-srt343/persian-rag-chatbot) |
| 🐛 **Report a Bug** | [Open an Issue](https://github.com/A-srt343/persian-rag-chatbot/issues/new) |
| 💡 **Request a Feature** | [Open a Discussion](https://github.com/A-srt343/persian-rag-chatbot/discussions) |
| 📖 **Documentation** | [Read the Docs](https://github.com/A-srt343/persian-rag-chatbot/wiki) |
| 📸 **Instagram** | [@project_srt343](https://instagram.com/project_srt343) |
| ⭐ **Star the Project** | [Give it a star](https://github.com/A-srt343/persian-rag-chatbot/stargazers) |

---

## ⚠️ Note About the API Key

This project **does not ship with an AI API key**. This decision was made for security and flexibility reasons:

- To prevent the developer's personal key from leaking into a public repository
- To allow each user to bring their own **free** API key
- So that API costs are paid directly by the user, not the developer

### 🧪 Testing

The project **has been fully tested before release** and performs its assigned tasks with good accuracy. Test results:

| Component | Status | Description |
|-----------|--------|-------------|
| PDF document loading | ✅ | Supports both Persian and English files |
| Accurate answering | ✅ | Answers are extracted only from the document content |
| Source display | ✅ | Page number and source snippet are shown |
| Hallucination prevention | ✅ | Model stays faithful to the document |
| Multilingual support | ✅ | Persian, Arabic, English |
| Response speed | ✅ | 1–3 seconds per question |

### 🔑 How to Enable

To run the project, create a **free** API key from [Groq](https://console.groq.com/keys) and place it in a `.env` file:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
