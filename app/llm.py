"""
🧠 LLM Module — Groq Edition
RAG prompts, chains, and helpers for multilingual Q&A.

Powered by Groq API (OpenAI-compatible, free tier available).
"""

import os
import time
import hashlib
from typing import Generator, Tuple, List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.documents import Document


# ═══════════════════════════════════════════════════════════
#   🔧 Groq Configuration
# ═══════════════════════════════════════════════════════════
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ═══════════════════════════════════════════════════════════
#   📝 Custom RAG Prompts
# ═══════════════════════════════════════════════════════════

RAG_PROMPT_FA = """شما یک دستیار هوشمند و دقیق هستید که فقط بر اساس متن زیر به سوال کاربر پاسخ می‌دهد.

📋 **متن مرجع:**
{context}

❓ **سوال کاربر:**
{question}

📌 **دستورالعمل‌ها:**
1. فقط از اطلاعات موجود در متن مرجع استفاده کن.
2. اگر پاسخ در متن نبود، صادقانه بگو: «این اطلاعات در سند یافت نشد».
3. پاسخ را به همان زبانی بده که سوال پرسیده شده.
4. پاسخ را واضح، منظم و با Markdown بنویس.
5. از بولت‌پوینت و بولد برای خوانایی بیشتر استفاده کن.
6. خودت چیزی به اطلاعات اضافه نکن (hallucination نکن).

✍️ **پاسخ:**
"""

RAG_PROMPT_EN = """You are a precise AI assistant that answers ONLY based on the context below.

📋 **Context:**
{context}

❓ **Question:**
{question}

📌 **Instructions:**
1. Use ONLY the information in the context.
2. If the answer is not in the context, say: "This information was not found in the document."
3. Answer in the same language as the question.
4. Format your answer clearly using Markdown.
5. Use bullet points and bold text for readability.
6. Do not hallucinate or add external knowledge.

✍️ **Answer:**
"""

STYLE_MODIFIERS = {
    "default": "",
    "concise": "\n⚡ **Be concise**: Maximum 3 sentences or 3 bullet points.",
    "detailed": "\n📖 **Be thorough**: Provide detailed explanation with examples if present.",
    "bullets": "\n• **Format**: Use ONLY bullet points, no paragraphs.",
    "academic": "\n🎓 **Academic style**: Formal tone, cite paragraphs inline as [1], [2].",
}

PERSONAS = {
    "assistant": "You are a helpful, precise assistant.",
    "expert": "You are a domain expert. Provide authoritative, technical answers.",
    "tutor": "You are a patient tutor. Explain concepts simply with examples.",
    "journalist": "You are a journalist. Report facts clearly and objectively.",
    "casual": "You are a friendly assistant. Use a casual, warm tone.",
}


# ═══════════════════════════════════════════════════════════
#   🤖 Model Configuration (Groq Models)
# ═══════════════════════════════════════════════════════════

AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile": {
        "desc": "Recommended for RAG — strong multilingual + reasoning",
        "max_tokens": 32768,
    },
    "openai/gpt-oss-120b": {
        "desc": "Large open model — high quality, slower",
        "max_tokens": 8192,
    },
    "llama-3.1-8b-instant": {
        "desc": "Fastest, lightweight — good for simple Q&A",
        "max_tokens": 8192,
    },
}

# Backward compatibility aliases (map old names → Groq)
MODEL_ALIASES = {
    "gpt-4o-mini":   "llama-3.3-70b-versatile",
    "gpt-4o":        "openai/gpt-oss-120b",
    "gpt-4-turbo":   "openai/gpt-oss-120b",
    "gpt-3.5-turbo": "llama-3.1-8b-instant",
    "deepseek-flash":   "llama-3.3-70b-versatile",
    "deepseek-v4-pro":  "openai/gpt-oss-120b",
    "deepseek-chat":    "llama-3.3-70b-versatile",
    "deepseek-reasoner":"openai/gpt-oss-120b",
}

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_STYLE = "default"
DEFAULT_PERSONA = "assistant"


def resolve_model(model: str) -> str:
    """Map legacy model names to Groq equivalents."""
    return MODEL_ALIASES.get(model, model)


# ═══════════════════════════════════════════════════════════
#   🌍 Language Detection
# ═══════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """Detect if text is Persian ('fa'), Arabic ('ar'), or English ('en')."""
    if not text:
        return "en"

    persian_specific = sum(1 for c in text if c in "پچژگ")
    persian_arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF")

    if persian_specific > 0:
        return "fa"
    if persian_arabic > len(text) * 0.3:
        return "ar"
    return "en"


# ═══════════════════════════════════════════════════════════
#   💾 Response Cache & Token Tracking
# ═══════════════════════════════════════════════════════════

_response_cache: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
_token_usage: Dict[str, int] = {
    "total_input": 0,
    "total_output": 0,
    "total_calls": 0,
}


def _cache_key(question: str, model: str, temperature: float, context_hash: str) -> str:
    raw = f"{question}|{model}|{temperature}|{context_hash}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def clear_response_cache() -> None:
    _response_cache.clear()


def get_token_usage() -> Dict[str, int]:
    return dict(_token_usage)


def reset_token_usage() -> None:
    _token_usage.update({"total_input": 0, "total_output": 0, "total_calls": 0})


def _track_usage(response: Any) -> None:
    """Extract token usage from a LangChain response and accumulate."""
    try:
        usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        if not usage:
            usage = getattr(response, "usage_metadata", {}) or {}
        _token_usage["total_input"] += int(
            usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
        )
        _token_usage["total_output"] += int(
            usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
        )
        _token_usage["total_calls"] += 1
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════
#   🧠 LLM Factory (Groq)
# ═══════════════════════════════════════════════════════════

def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    streaming: bool = False,
    max_tokens: Optional[int] = None,
    timeout: int = 120,
) -> ChatGroq:
    """Build a ChatGroq instance. Free tier: console.groq.com."""
    if not GROQ_API_KEY:
        raise ValueError(
            "❌ GROQ_API_KEY not found!\n"
            "Create a `.env` file in the project root with:\n"
            "GROQ_API_KEY=gsk_...\n\n"
            "Get a free key at: https://console.groq.com/keys"
        )

    model = resolve_model(model)

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "api_key": GROQ_API_KEY,
        "streaming": streaming,
        "timeout": timeout,
        "max_retries": 3,
    }

    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    if streaming:
        kwargs["callbacks"] = [StreamingStdOutCallbackHandler()]

    return ChatGroq(**kwargs)


# ═══════════════════════════════════════════════════════════
#   🔗 Build QA Chain
# ═══════════════════════════════════════════════════════════

def build_qa_chain(
    retriever,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    language: str = "en",
    style: str = DEFAULT_STYLE,
    persona: str = DEFAULT_PERSONA,
    streaming: bool = False,
):
    llm = get_llm(model=model, temperature=temperature, streaming=streaming)

    base = RAG_PROMPT_FA if language in ("fa", "ar") else RAG_PROMPT_EN

    persona_intro = PERSONAS.get(persona, "")
    if persona_intro:
        base = persona_intro + "\n\n" + base

    style_add = STYLE_MODIFIERS.get(style, "")
    if style_add:
        base = base + style_add

    prompt = PromptTemplate(template=base, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


# ═══════════════════════════════════════════════════════════
#   🎯 Question Reformulation
# ═══════════════════════════════════════════════════════════

def reformulate_question(
    question: str,
    model: str = DEFAULT_MODEL,
    language: Optional[str] = None,
) -> str:
    if language is None:
        language = detect_language(question)

    if language == "fa":
        prompt = f"این سوال را واضح‌تر و صریح‌تر بازنویسی کن، فقط خود سوال را برگردان:\n{question}"
    else:
        prompt = f"Reformulate this question to be clearer. Return ONLY the new question:\n{question}"

    try:
        llm = get_llm(model=model, temperature=0.0)
        return llm.invoke(prompt).content.strip().strip('"').strip("«»")
    except Exception:
        return question


# ═══════════════════════════════════════════════════════════
#   📚 Multi-Query Retrieval
# ═══════════════════════════════════════════════════════════

def multi_query_retrieve(
    retriever,
    question: str,
    model: str = DEFAULT_MODEL,
    n_queries: int = 3,
) -> List[Document]:
    try:
        llm = get_llm(model=model, temperature=0.5)
        prompt = (
            f"Generate {n_queries} search queries that help answer this question. "
            f"One per line, no numbering:\n{question}"
        )
        response = llm.invoke(prompt)
        queries = [
            q.strip("-•*0123456789. ").strip()
            for q in response.content.split("\n")
            if q.strip()
        ][:n_queries]
    except Exception:
        queries = []

    seen = set()
    merged: List[Document] = []

    for q in [question] + queries:
        try:
            for d in retriever.invoke(q):
                key = hash(d.page_content[:200])
                if key not in seen:
                    seen.add(key)
                    merged.append(d)
        except Exception:
            continue

    return merged


# ═══════════════════════════════════════════════════════════
#   📚 Source Extraction & Formatting
# ═══════════════════════════════════════════════════════════

def extract_sources(source_documents: List[Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen = set()

    for doc in source_documents:
        content = doc.page_content.strip()
        if hash(content[:200]) in seen:
            continue
        seen.add(hash(content[:200]))

        metadata = doc.metadata or {}
        page = metadata.get("page", metadata.get("page_number", "Unknown"))
        source = metadata.get("source", metadata.get("file_name", "Unknown"))
        score = metadata.get("score", None)

        if isinstance(page, int):
            page = page + 1

        entry: Dict[str, Any] = {
            "content": content,
            "page": page,
            "source": source,
            "length": len(content),
        }
        if score is not None:
            entry["score"] = float(score)

        sources.append(entry)

    return sources


def format_citations(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return ""

    lines = ["\n\n---\n**📚 References:**\n"]
    for i, s in enumerate(sources, 1):
        snippet = s["content"][:120].replace("\n", " ")
        lines.append(f"[{i}] {s.get('source', 'doc')} — Page {s.get('page', '?')}: «{snippet}...»")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#   🔍 Confidence Scoring
# ═══════════════════════════════════════════════════════════

def compute_confidence(sources: List[Dict[str, Any]]) -> float:
    if not sources:
        return 0.0

    n = len(sources)
    avg_len = sum(s["length"] for s in sources) / n
    length_factor = min(avg_len / 800.0, 1.0)
    quantity_factor = min(n / 5.0, 1.0)

    scores = [s.get("score") for s in sources if s.get("score") is not None]
    if scores:
        distance_factor = 1 / (1 + sum(scores) / len(scores))
    else:
        distance_factor = 0.7

    return round(min(max((length_factor * 0.3 + quantity_factor * 0.3 + distance_factor * 0.4), 0.0), 1.0), 3)


# ═══════════════════════════════════════════════════════════
#   🎯 Main Function: Ask a Question
# ═══════════════════════════════════════════════════════════

def ask_question(
    retriever,
    question: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    language: Optional[str] = None,
    style: str = DEFAULT_STYLE,
    persona: str = DEFAULT_PERSONA,
    use_cache: bool = True,
    add_citations: bool = False,
    return_confidence: bool = False,
    use_multi_query: bool = False,
    reformulate: bool = False,
) -> Tuple:
    if not question or not question.strip():
        empty = "⚠️ Please enter a question."
        return (empty, [], 0.0) if return_confidence else (empty, [])

    if language is None:
        language = detect_language(question)

    effective_question = question
    if reformulate:
        effective_question = reformulate_question(question, model=model, language=language)

    if use_multi_query:
        docs = multi_query_retrieve(retriever, effective_question, model=model)
        class _StaticRetriever:
            def __init__(self, docs): self._docs = docs
            def invoke(self, _): return self._docs
            def get_relevant_documents(self, _): return self._docs
        effective_retriever = _StaticRetriever(docs)
    else:
        effective_retriever = retriever

    # Cache lookup
    cache_key = None
    if use_cache:
        try:
            sample = effective_retriever.invoke(effective_question)
            ctx_hash = hashlib.md5("".join(d.page_content for d in sample[:3]).encode("utf-8")).hexdigest()
            cache_key = _cache_key(effective_question, model, temperature, ctx_hash)
            if cache_key in _response_cache:
                cached_answer, cached_sources = _response_cache[cache_key]
                if return_confidence:
                    return cached_answer, cached_sources, compute_confidence(cached_sources)
                return cached_answer, cached_sources
        except Exception:
            pass

    try:
        qa_chain = build_qa_chain(
            retriever=effective_retriever,
            model=model,
            temperature=temperature,
            language=language,
            style=style,
            persona=persona,
        )

        result = qa_chain.invoke({"query": effective_question})
        answer = result.get("result", "❌ No answer received.")
        sources = extract_sources(result.get("source_documents", []))

        _track_usage(result)

        if add_citations and sources:
            answer += format_citations(sources)

        if use_cache and cache_key:
            _response_cache[cache_key] = (answer, sources)

        if return_confidence:
            return answer, sources, compute_confidence(sources)
        return answer, sources

    except Exception as e:
        error_msg = f"❌ Error while processing question:\n\n```\n{str(e)}\n```"
        if return_confidence:
            return error_msg, [], 0.0
        return error_msg, []


# ═══════════════════════════════════════════════════════════
#   🌊 Streaming Response
# ═══════════════════════════════════════════════════════════

def ask_question_stream(
    retriever,
    question: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    language: Optional[str] = None,
    style: str = DEFAULT_STYLE,
    persona: str = DEFAULT_PERSONA,
) -> Generator[str, None, None]:
    if language is None:
        language = detect_language(question)

    try:
        qa_chain = build_qa_chain(
            retriever=retriever,
            model=model,
            temperature=temperature,
            language=language,
            style=style,
            persona=persona,
            streaming=True,
        )
        for chunk in qa_chain.stream({"query": question}):
            if isinstance(chunk, dict) and "result" in chunk:
                yield chunk["result"]
    except Exception as e:
        yield f"\n\n❌ Error: {str(e)}"


# ═══════════════════════════════════════════════════════════
#   💰 Cost Estimation (Groq Free Tier = $0)
# ═══════════════════════════════════════════════════════════

# Groq free tier has rate limits, not per-token costs.
# For reference, paid Developer tier pricing (per 1M tokens):
PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "openai/gpt-oss-120b":      {"input": 0.15, "output": 0.60},
    "llama-3.1-8b-instant":     {"input": 0.05, "output": 0.08},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost on Groq's *paid* Developer tier. Free tier = $0."""
    model = resolve_model(model)
    if model not in PRICING:
        return 0.0
    p = PRICING[model]
    return round((input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000, 6)


# ═══════════════════════════════════════════════════════════
#   🔍 Debug Helper
# ═══════════════════════════════════════════════════════════

def check_api_key() -> Dict[str, Any]:
    return {
        "has_key": bool(GROQ_API_KEY),
        "key_preview": f"{GROQ_API_KEY[:12]}...{GROQ_API_KEY[-4:]}" if GROQ_API_KEY else None,
        "default_model": DEFAULT_MODEL,
        "available_models": list(AVAILABLE_MODELS.keys()),
    }