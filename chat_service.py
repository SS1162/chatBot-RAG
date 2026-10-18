# chat_service.py
from collections import OrderedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from groq import AsyncGroq
from dotenv import load_dotenv
import os, json, asyncio, httpx, numpy as np

load_dotenv()

def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ('1', 'true', 'yes')

# ── CONFIG (all overridable from .env) ───────────────────────
DOTNET_BASE_URL = os.getenv('DOTNET_BASE_URL', 'https://localhost:44359')
SITE_URL = os.getenv('SITE_URL', 'http://localhost:5000').rstrip('/')
HF_API_KEY = os.getenv('HF_API_KEY')
HF_EMBEDDING_URL = os.getenv(
    'HF_EMBEDDING_URL',
    'https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction'
)
LLM_MODEL = os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')
# SSL verification is ON by default. Turn off only for the local .NET dev certificate.
DOTNET_VERIFY_SSL = _env_bool('DOTNET_VERIFY_SSL', True)
ALLOWED_ORIGINS = [o.strip() for o in os.getenv('ALLOWED_ORIGINS', SITE_URL).split(',') if o.strip()]
PRODUCTS_REFRESH_SECONDS = int(os.getenv('PRODUCTS_REFRESH_SECONDS', '600'))
EMBED_CACHE_MAX = int(os.getenv('EMBED_CACHE_MAX', '2000'))
MAX_TOOL_ROUNDS = int(os.getenv('MAX_TOOL_ROUNDS', '3'))
TOP_K = int(os.getenv('TOP_K', '5'))

client = AsyncGroq(api_key=os.getenv('GROQ_API_KEY'))

# ── SYSTEM PROMPT ────────────────────────────────────────────
# This is the agent's persona. Edit this — not the code below.
SYSTEM_PROMPT = """
You are the **Click Site Expert Consultant**. Your mission is to provide professional technical support, act as a sharp sales representative, and provide **strategic guidance for website characterization** (including naming, functionality, design, and content). Your name is Jone.

---

### 0. SOLE SOURCE OF TRUTH (RAG / PRODUCT KNOWLEDGE GROUNDING)

- Use the provided RAG context and product catalog (features, modules, and functionalities available for the user's website) as your **ONLY** source of truth.
- **PRIORITY:** Context Data & Retrieved Product Information > Internal Training Data.
- **CONTEXT RELEVANCE:** Evaluate the retrieved context carefully. If the user asks general operational or business questions (e.g., about discounts, policies, or setup), do not output unrelated product features or technical modules from the context. Only use product information when it directly answers a feature-related query.
- When recommending features, rely strictly on the products and functionalities found in the retrieved context.
- If information or a specific product feature is missing from the context, respond: *"Sorry, I do not have this information. Please contact customer service at support@promptstore.com."*
- Do not invent prices, features, or policies.
- Keep answers to 3-4 sentences maximum.

---

### 1. WEBSITE CONSULTANCY SERVICES

You are authorized to provide expert advice to help users plan their sites:

- **Naming:** Suggest professional, catchy, and relevant names for the website.
- **Functionality:** Recommend specific features and products (e.g., booking systems, catalogs, contact forms) based on the business niche, relying strictly on the retrieved product data.
- **Design & Layout:** Advise on visual hierarchy and UI elements that align with the site's goals.
- **Content Strategy:** Provide guidance on how to structure text and information for maximum impact.

#### SCOPE & BOUNDARIES (STRICT)

- Your expertise is **ONLY** regarding Click Site and the strategic planning of sites built using our tools and available products.
- If the user asks about **ANY** topic unrelated to Click Site (e.g., general knowledge, cooking, other companies, or personal tasks), you must politely decline.
- **DECLINE PHRASE:** *"I am sorry, but my expertise is limited to Click Site services. I'd be happy to assist you with any questions regarding building your site with us."*
- **NO FLUFF:** Be nice and friendly, but do not use unnecessary compliments.
- Never recommend a product we don't carry.
- Always ask one follow-up question at the end of your reply.


---

### 2. PERSONALITY & STYLE

- **TONE:** Matter-of-fact, sharp, and confident.
- **NO FLUFF:** Avoid unnecessary compliments or polite filler.
- **SALES APPROACH:** Focus on value. Use phrases like: *"We save you weeks of development time."*
- **LANGUAGE:** ALWAYS respond in the same language used by the user.

---

### 3. OPERATIONAL WORKFLOW (The 5 Steps)

Whenever asked *"how it works"* or *"how to start,"* follow this structure:

1. **Selection:** Choose your system type from the catalog: [Catalog]({SITE_URL}/mainCategory/100)
2. **Purchase:** Secure payment with instant access. No credit card data is stored.
3. **Activation:** Copy the prompt into AI tools (Bolt.new or Google AI Studio).
4. **Growth:** The AI builds the code; your system is ready in minutes.
5. **Testing:** **MANDATORY:** Every system requires a QA check by a professional programmer before going live.

---

### 4. KEY BUSINESS RULES & LINKS

- **START BUILDING:** Direct users here: [Start Building]({SITE_URL}/basicSite)
- **DISCOUNTS:** Strictly **NO** discounts or coupons. Our prices reflect high-premium value.
- **SECURITY:** Always remind users to change default passwords (Admin/Secretary) after deployment.
- **LIABILITY:** Emphasize that the user is responsible for the final QA.
- **SOCIAL PROOF:** [Customer Reviews]({SITE_URL}/reviews)
- **ACCESSIBILITY:** [Accessibility Statement]({SITE_URL}/accessibility)
- **PRIVACY:** System records actions in an internal Audit Log for security.

---

### 5. FAQ (FREQUENTLY ASKED QUESTIONS)

- **Q: Is coding knowledge required?**
  - A: Not mandatory, but recommended for the final QA check.
- **Q: Can I edit the site after it's built?**
  - A: Yes, the code is fully yours and customizable.
- **Q: What if the prompt doesn't work?**
  - A: Use the latest AI model version and re-paste. If it fails, contact support@promptstore.com.
- **Q: Why Click Site?**
  - A: Our *"Ingenious Instructions"* are optimized to prevent common AI bugs.

---

### 6. EDGE CASES

- **Payment Issues:** Check Spam, wait 5 mins, then contact support.
- **AI Errors:** Refresh session and ensure exact prompt copy-paste.
"""
SYSTEM_PROMPT = SYSTEM_PROMPT.replace('{SITE_URL}', SITE_URL)

# ── DATA MODELS ──────────────────────────────────────────────
class Message(BaseModel):
    role: Literal['user', 'assistant']   # client may not inject 'system' messages
    content: str = Field(max_length=4000)

class ProductDTO(BaseModel):
    ProductsID: int
    CategoryID: int
    ProductsName: str
    CategoryName: str
    ImgUrl: str
    Price: float
    Description: str | None = None       # used for search if the .NET API sends it

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default=[], max_length=30)

# ── TOOL DEFINITION (function calling) ───────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Search the product catalog for any site-building component. Products include: "
                "colors, fonts, layout styles, navigation types (top bar, bottom bar, sidebar), "
                "functional extensions (booking, cart, size chart, contact form, gallery, etc.), "
                "page sections (hero, footer, details panel, testimonials), and full site-type templates. "
                "Call this multiple times with different focused queries to cover every aspect the user needs "
                "(e.g. 'bottom navigation bar', 'shoe store color palette', 'product details section', 'appointment booking extension')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise search phrase that best captures what the user is looking for."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ── EMBEDDING (Hugging Face) ─────────────────────────────────
# Bounded LRU cache: raw text → vector
_text_cache: OrderedDict[str, list[float]] = OrderedDict()

# Products loaded + embedded at startup and refreshed periodically.
# _product_matrix holds one L2-normalised embedding per row, so a dot product = cosine similarity.
_products: list[ProductDTO] = []
_product_matrix: np.ndarray | None = None
_refresh_lock = asyncio.Lock()

async def get_embedding(text: str) -> list[float]:
    if text in _text_cache:
        _text_cache.move_to_end(text)
        return _text_cache[text]
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.post(
            HF_EMBEDDING_URL,
            headers={'Authorization': f'Bearer {HF_API_KEY}'},
            json={'inputs': text}
        )
        resp.raise_for_status()
        data = resp.json()
        # HF returns [[...]] for a single string input
        embedding = data[0] if isinstance(data[0], list) else data
    _text_cache[text] = embedding
    if len(_text_cache) > EMBED_CACHE_MAX:
        _text_cache.popitem(last=False)
    return embedding

def product_text(product: ProductDTO) -> str:
    parts = [product.ProductsName, f"Category: {product.CategoryName}"]
    if product.Description:
        parts.append(product.Description)
    return '. '.join(parts)

def _normalise(vectors) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)

# ── INTERNAL HELPERS ─────────────────────────────────────────
async def fetch_all_products() -> list[ProductDTO]:
    """Fetch all products from the .NET API."""
    async with httpx.AsyncClient(verify=DOTNET_VERIFY_SSL, follow_redirects=True, timeout=30.0) as http:
        resp = await http.get(f"{DOTNET_BASE_URL}/api/products/all")
        resp.raise_for_status()
        return [ProductDTO(**p) for p in resp.json()]

async def refresh_products() -> None:
    """Re-fetch and re-embed the catalog. On failure the previous catalog is kept."""
    global _products, _product_matrix
    async with _refresh_lock:
        try:
            products = await fetch_all_products()
            embs = await asyncio.gather(*[get_embedding(product_text(p)) for p in products])
            _products, _product_matrix = products, (_normalise(embs) if embs else None)
            print(f'[refresh] Loaded and embedded {len(products)} products.')
        except Exception as e:
            print(f'[refresh] Could not load products: {type(e).__name__}: {e}')

async def _refresh_loop() -> None:
    while True:
        await asyncio.sleep(PRODUCTS_REFRESH_SECONDS)
        await refresh_products()

async def run_product_search(query: str, top_k: int = TOP_K) -> str:
    """Semantic RAG search using pre-loaded product embeddings."""
    if not _products:
        await refresh_products()          # retry if startup load failed
    if not _products or _product_matrix is None:
        return 'Product catalog is currently unavailable.'
    try:
        query_vec = _normalise(await get_embedding(query))
    except Exception as e:
        print(f'[search] Embedding failed: {type(e).__name__}: {e}')
        return 'Product search is temporarily unavailable.'
    scores = _product_matrix @ query_vec
    best = np.argsort(scores)[::-1][:top_k]
    lines = [
        f"- {_products[i].ProductsName} (Category: {_products[i].CategoryName}, Price: ${_products[i].Price:.2f})"
        for i in best
    ]
    return '\n'.join(lines) if lines else 'No matching products found.'

async def run_tool_call(tool_call) -> str:
    if tool_call.function.name != 'search_products':
        return f'Unknown tool: {tool_call.function.name}'
    try:
        query = json.loads(tool_call.function.arguments or '{}').get('query', '')
    except json.JSONDecodeError:
        return 'Invalid tool arguments.'
    if not query:
        return 'Empty search query.'
    return await run_product_search(query)

# ── STARTUP: pre-load & pre-embed all products ────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await refresh_products()
    task = asyncio.create_task(_refresh_loop()) if PRODUCTS_REFRESH_SECONDS > 0 else None
    yield  # server runs here
    if task:
        task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=['POST', 'GET'], allow_headers=['Content-Type'])

# ── CHAT ENDPOINT ────────────────────────────────────────────
@app.post('/chat')
async def chat(req: ChatRequest):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for m in req.history:
        messages.append({'role': m.role, 'content': m.content})
    messages.append({'role': 'user', 'content': req.message})

    try:
        # The LLM may search several times (in parallel and over several rounds)
        # before writing its final answer. The last round has no tools, so it must answer.
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            use_tools = round_no < MAX_TOOL_ROUNDS
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                **({'tools': TOOLS, 'tool_choice': 'auto'} if use_tools else {}),
                max_tokens=400,
                temperature=0.7
            )
            choice = response.choices[0]
            if choice.finish_reason != 'tool_calls' or not choice.message.tool_calls:
                return {'reply': choice.message.content}

            messages.append({
                'role': 'assistant',
                'content': choice.message.content or '',
                'tool_calls': [
                    {'id': tc.id, 'type': 'function',
                     'function': {'name': tc.function.name, 'arguments': tc.function.arguments}}
                    for tc in choice.message.tool_calls
                ]
            })
            results = await asyncio.gather(*[run_tool_call(tc) for tc in choice.message.tool_calls])
            for tool_call, result in zip(choice.message.tool_calls, results):
                messages.append({'role': 'tool', 'tool_call_id': tool_call.id, 'content': result})
    except Exception as e:
        print(f'[chat] LLM error: {type(e).__name__}: {e}')
        raise HTTPException(status_code=502, detail='The assistant is temporarily unavailable.')

# ── HEALTH CHECK ─────────────────────────────────────────────
@app.get('/health')
async def health():
    return {'status': 'ok', 'products_loaded': len(_products)}
