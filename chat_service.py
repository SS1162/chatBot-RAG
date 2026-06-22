# chat_service.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os, json, httpx, numpy as np

load_dotenv()
DOTNET_BASE_URL = os.getenv('DOTNET_BASE_URL', 'https://localhost:44359')
HF_API_KEY = os.getenv('HF_API_KEY')
HF_EMBEDDING_URL = 'https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2'
client = Groq(api_key=os.getenv('GROQ_API_KEY'), http_client=httpx.Client(verify=False))

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

1. **Selection:** Choose your system type from the catalog: [Catalog](http://localhost:5000/mainCategory/100)
2. **Purchase:** Secure payment with instant access. No credit card data is stored.
3. **Activation:** Copy the prompt into AI tools (Bolt.new or Google AI Studio).
4. **Growth:** The AI builds the code; your system is ready in minutes.
5. **Testing:** **MANDATORY:** Every system requires a QA check by a professional programmer before going live.

---

### 4. KEY BUSINESS RULES & LINKS

- **START BUILDING:** Direct users here: [Start Building](http://localhost:5000/basicSite)
- **DISCOUNTS:** Strictly **NO** discounts or coupons. Our prices reflect high-premium value.
- **SECURITY:** Always remind users to change default passwords (Admin/Secretary) after deployment.
- **LIABILITY:** Emphasize that the user is responsible for the final QA.
- **SOCIAL PROOF:** [Customer Reviews](http://localhost:5000/reviews)
- **ACCESSIBILITY:** [Accessibility Statement](http://localhost:5000/accessibility)
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

# ── DATA MODELS ──────────────────────────────────────────────
class Message(BaseModel):
    role: str      # 'user' or 'assistant'
    content: str

class ProductDTO(BaseModel):
    ProductsID: int
    CategoryID: int
    ProductsName: str
    CategoryName: str
    ImgUrl: str
    Price: float

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []

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

import asyncio

# ── EMBEDDING (Hugging Face) ─────────────────────────────────
_text_cache: dict[str, list[float]] = {}     # cache raw text → vector

# Products loaded + embedded at startup
_products: list[ProductDTO] = []
_product_embs: list[list[float]] = []

async def get_embedding(text: str) -> list[float]:
    if text in _text_cache:
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
        return embedding

async def get_product_embedding(product: ProductDTO) -> list[float]:
    text = f"{product.ProductsName} {product.CategoryName}".strip()
    return await get_embedding(text)

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ── INTERNAL HELPERS ─────────────────────────────────────────
async def fetch_all_products() -> list[ProductDTO]:
    """Fetch all products from the .NET API."""
    async with httpx.AsyncClient(verify=False, follow_redirects=True) as http:
        resp = await http.get(f"{DOTNET_BASE_URL}/api/products/all")
        resp.raise_for_status()
        return [ProductDTO(**p) for p in resp.json()]

async def run_product_search(query: str, top_k: int = 5) -> str:
    """Semantic RAG search using pre-loaded product embeddings."""
    if not _products:
        return 'No products available.'
    query_emb = await get_embedding(query)
    scored = sorted(
        zip(_product_embs, _products),
        key=lambda x: cosine_similarity(query_emb, x[0]),
        reverse=True
    )
    lines = [
        f"- {p.ProductsName} (Category: {p.CategoryName}, Price: ${p.Price:.2f})"
        for _, p in scored[:top_k]
    ]
    return '\n'.join(lines) if lines else 'No matching products found.'

# ── STARTUP: pre-load & pre-embed all products ────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _products, _product_embs
    try:
        _products = await fetch_all_products()
        _product_embs = await asyncio.gather(*[get_product_embedding(p) for p in _products])
        print(f'[startup] Loaded and embedded {len(_products)} products.')
    except Exception as e:
        print(f'[startup] Could not pre-load products: {type(e).__name__}: {e}')
    yield  # server runs here

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

# ── CHAT ENDPOINT ────────────────────────────────────────────
@app.post('/chat')
async def chat(req: ChatRequest):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for m in req.history:
        messages.append({'role': m.role, 'content': m.content})
    messages.append({'role': 'user', 'content': req.message})

    # 1. First LLM call — LLM decides whether (and what) to search
    first_response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=messages,
        tools=TOOLS,
        tool_choice='auto',
        max_tokens=400,
        temperature=0.7
    )

    choice = first_response.choices[0]

    # 2. No tool needed → return directly
    if choice.finish_reason != 'tool_calls':
        return {'reply': choice.message.content}

    # 3. LLM called search_products (possibly multiple times in parallel)
    #    e.g. query="warm shoe colors", query="rounded font", query="size chart extension"
    messages.append(choice.message)

    search_results = await asyncio.gather(*[
        run_product_search(json.loads(tc.function.arguments)['query'])
        for tc in choice.message.tool_calls
    ])


    for tool_call, result in zip(choice.message.tool_calls, search_results):
        messages.append({
            'role': 'tool',
            'tool_call_id': tool_call.id,
            'content': result
        })

    # 4. Second LLM call — sees all search results, writes the final reply
    final_response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=messages,
        max_tokens=400,
        temperature=0.7
    )
    return {'reply': final_response.choices[0].message.content}
