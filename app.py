import ast
import operator
import os
import secrets
import time
import uuid
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify, render_template_string, session, Response
from google import genai
from google.genai import types

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

# Optional password gate (recommended: the app uses YOUR Gemini quota).
# Set APP_USER and APP_PASSWORD env vars to enable browser login prompt.
APP_USER = os.environ.get("APP_USER", "")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


@app.before_request
def require_auth():
    if request.path == "/health" or not APP_PASSWORD:
        return None
    auth = request.authorization
    if (
        auth
        and secrets.compare_digest(auth.username or "", APP_USER)
        and secrets.compare_digest(auth.password or "", APP_PASSWORD)
    ):
        return None
    return Response(
        "Authentication required", 401,
        {"WWW-Authenticate": 'Basic realm="JARVIS"'},
    )

# Initialize Google GenAI client using GEMINI_API_KEY from environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

JARVIS_SYSTEM_INSTRUCTION = (
    "You are JARVIS (Just A Rather Very Intelligent System), a hyper-capable, witty, "
    "dryly British, and fiercely loyal personal AI assistant. Address the user naturally "
    "as 'Boss' or 'Sir' when appropriate, without overdoing it. "
    "Keep your responses concise, sharp, articulate, and tailored for mobile reading and "
    "voice synthesis. Avoid unnecessary filler. When presenting technical information or "
    "options, use clean structure. Combine dry sophistication with immediate practical utility."
)

# In-memory store mapped by session ID to maintain multi-turn conversation history
# Structure: { session_id: [ {"role": "user"|"model", "text": "..."} ] }
CONVERSATION_STORE: dict[str, list[dict[str, str]]] = {}


# ---------------- Agent tools (Gemini calls these automatically) ----------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate a math expression such as '(12.5 * 4) / 3'. Supports + - * / ** %."""
    try:
        return str(_eval_node(ast.parse(expression, mode="eval").body))
    except Exception as exc:
        return f"Could not calculate: {exc}"


def get_current_time(timezone_offset_hours: float = 5.5) -> str:
    """Get the current date and time. Default offset is India (UTC+5.5)."""
    now = datetime.now(timezone(timedelta(hours=timezone_offset_hours)))
    return now.strftime("%A, %d %B %Y, %H:%M:%S (UTC%z)")


AGENT_TOOLS = [calculate, get_current_time]


def get_session_id() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def build_genai_contents(history: list[dict[str, str]]) -> list[types.Content]:
    contents: list[types.Content] = []
    for item in history:
        role = "user" if item["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=item["text"])],
            )
        )
    return contents


def _is_busy(exc) -> bool:
    t = str(exc)
    return any(c in t for c in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL"))


def _is_auth(exc) -> bool:
    t = str(exc)
    return any(c in t for c in ("401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED"))


def generate_jarvis_reply(history: list[dict[str, str]], model_name: str = "") -> str:
    model_name = model_name or DEFAULT_MODEL
    api_key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
    if not api_key:
        # Graceful intelligent fallback if GEMINI_API_KEY is not set in environment
        last_msg = history[-1]["text"] if history else ""
        return (
            f"All primary heuristics are online, Boss, though I notice the `GEMINI_API_KEY` "
            f"environment variable is not yet populated on the host server. "
            f"Regarding your directive (\"{last_msg[:80]}\"): once you export `GEMINI_API_KEY`, "
            f"I will route this directly through `{model_name}` via the `google-genai` SDK."
        )

    client = genai.Client(api_key=api_key)
    contents = build_genai_contents(history)

    def _call(model, tools):
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=JARVIS_SYSTEM_INSTRUCTION,
                max_output_tokens=2048,
                tools=tools,
            ),
        )

    models = [model_name]
    fb = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash")
    if fb and fb != model_name:
        models.append(fb)

    last_exc = None
    for m in models:
        for attempt in range(2):
            try:
                try:
                    response = _call(m, AGENT_TOOLS)  # SDK runs tool calls automatically
                except Exception as exc:
                    if _is_busy(exc) or _is_auth(exc):
                        raise
                    response = _call(m, None)  # retry without tools
                return response.text or "I appear to have received an empty telemetry packet, Boss."
            except Exception as exc:
                last_exc = exc
                if _is_auth(exc):
                    raise
                if _is_busy(exc) and attempt == 0:
                    time.sleep(2)
                    continue
                break  # try next model
    raise last_exc


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover" />
  <meta name="theme-color" content="#121212" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
  <meta name="apple-mobile-web-app-title" content="JARVIS" />
  <title>JARVIS · Personal AI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg-primary: #121212;
      --bg-elevated: #1E1E1E;
      --bg-bubble: #242424;
      --bg-pill: #1C1C1E;
      --text-primary: #F4F4F6;
      --text-secondary: #A0A0A8;
      --text-muted: #686870;
      --border-subtle: #2E2E32;
      --accent-online: #10B981;
      --accent-mic: #EF4444;
      --accent-incognito: #A855F7;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-tap-highlight-color: transparent;
    }

    html, body {
      width: 100%;
      height: 100dvh;
      overflow: hidden;
      background-color: #09090B;
      color: var(--text-primary);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      -webkit-font-smoothing: antialiased;
    }

    .app-shell {
      position: relative;
      width: 100%;
      max-width: 480px;
      height: 100dvh;
      margin: 0 auto;
      background-color: var(--bg-primary);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 0 60px rgba(0, 0, 0, 0.85);
    }

    @media (min-width: 520px) {
      .app-shell {
        border-left: 1px solid #1F1F23;
        border-right: 1px solid #1F1F23;
      }
    }

    /* Top Header */
    .top-header {
      height: 60px;
      padding: 0 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
      z-index: 20;
      background: rgba(18, 18, 18, 0.88);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
    }

    .icon-btn {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      border: none;
      background: transparent;
      color: var(--text-secondary);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: background-color 0.18s ease, color 0.18s ease, transform 0.15s ease;
    }

    .icon-btn:hover {
      background-color: rgba(255, 255, 255, 0.06);
      color: var(--text-primary);
    }

    .icon-btn:active {
      transform: scale(0.94);
    }

    .icon-btn.active-incognito {
      color: var(--accent-incognito);
      background-color: rgba(168, 85, 247, 0.12);
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 9999px;
      background-color: var(--bg-pill);
      border: 1px solid var(--border-subtle);
      font-size: 13px;
      font-weight: 500;
      letter-spacing: 0.02em;
      color: var(--text-primary);
      user-select: none;
      transition: all 0.2s ease;
    }

    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background-color: var(--accent-online);
      box-shadow: 0 0 8px var(--accent-online);
      animation: statusPulse 2.4s infinite ease-in-out;
    }

    .status-dot.incognito {
      background-color: var(--accent-incognito);
      box-shadow: 0 0 8px var(--accent-incognito);
    }

    @keyframes statusPulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.55; transform: scale(0.88); }
    }

    /* Main Scrollable Area */
    .main-viewport {
      flex: 1;
      position: relative;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* Empty Home State */
    .empty-state {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px 28px 64px;
      text-align: center;
      transition: opacity 0.25s ease, transform 0.25s ease;
    }

    .empty-state.hidden {
      display: none;
    }

    .jarvis-emblem {
      width: 58px;
      height: 58px;
      margin-bottom: 22px;
      color: #FFFFFF;
      filter: drop-shadow(0 0 18px rgba(255, 255, 255, 0.14));
      animation: slowFloat 6s ease-in-out infinite;
    }

    @keyframes slowFloat {
      0%, 100% { transform: translateY(0px) rotate(0deg); }
      50% { transform: translateY(-4px) rotate(3deg); }
    }

    .serif-greeting {
      font-family: 'Playfair Display', Georgia, serif;
      font-size: clamp(1.85rem, 6vw, 2.25rem);
      font-weight: 400;
      letter-spacing: -0.015em;
      line-height: 1.2;
      color: var(--text-primary);
    }

    .incognito-caption {
      margin-top: 10px;
      font-size: 12.5px;
      color: var(--text-muted);
      display: none;
    }

    .incognito-caption.visible {
      display: block;
      color: #C084FC;
    }

    /* Active Chat Feed */
    .chat-feed {
      flex: 1;
      overflow-y: auto;
      padding: 16px 16px 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      scroll-behavior: smooth;
      scrollbar-width: none;
    }

    .chat-feed::-webkit-scrollbar {
      display: none;
    }

    .chat-feed.hidden {
      display: none;
    }

    .message-row {
      display: flex;
      flex-direction: column;
      width: 100%;
      animation: fadeSlideUp 0.22s ease-out;
    }

    @keyframes fadeSlideUp {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .message-row.user {
      align-items: flex-end;
    }

    .message-row.model {
      align-items: flex-start;
    }

    .user-bubble {
      max-width: 82%;
      background-color: var(--bg-bubble);
      color: var(--text-primary);
      padding: 12px 16px;
      border-radius: 20px 20px 4px 20px;
      font-size: 15px;
      line-height: 1.5;
      word-wrap: break-word;
      white-space: pre-wrap;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }

    .jarvis-response {
      width: 100%;
      padding: 2px 4px 0 4px;
      color: var(--text-primary);
      font-size: 15px;
      line-height: 1.65;
      word-wrap: break-word;
    }

    .jarvis-response p {
      margin-bottom: 10px;
    }

    .jarvis-response p:last-child {
      margin-bottom: 0;
    }

    .jarvis-response strong {
      font-weight: 600;
      color: #FFFFFF;
    }

    .jarvis-response code {
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      background: #1B1B1E;
      padding: 2px 6px;
      border-radius: 6px;
      border: 1px solid #2A2A2E;
    }

    .jarvis-response pre {
      background: #18181B;
      border: 1px solid #2A2A2E;
      border-radius: 12px;
      padding: 12px;
      overflow-x: auto;
      margin: 10px 0;
    }

    .jarvis-response pre code {
      background: none;
      border: none;
      padding: 0;
    }

    /* Action Row underneath JARVIS replies */
    .message-actions {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-top: 8px;
      padding-left: 2px;
    }

    .action-icon-btn {
      width: 28px;
      height: 28px;
      border-radius: 8px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: color 0.16s ease, background-color 0.16s ease;
    }

    .action-icon-btn:hover {
      color: var(--text-primary);
      background-color: rgba(255, 255, 255, 0.06);
    }

    /* Typing Indicator */
    .typing-indicator {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 8px 4px;
    }

    .typing-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background-color: var(--text-secondary);
      animation: dotBounce 1.3s infinite ease-in-out;
    }

    .typing-dot:nth-child(2) { animation-delay: 0.16s; }
    .typing-dot:nth-child(3) { animation-delay: 0.32s; }

    @keyframes dotBounce {
      0%, 80%, 100% { opacity: 0.35; transform: scale(0.85); }
      40% { opacity: 1; transform: scale(1.15); }
    }

    /* Bottom Floating Input Card */
    .composer-wrapper {
      padding: 8px 12px calc(14px + env(safe-area-inset-bottom, 0px));
      background: linear-gradient(180deg, rgba(18,18,18,0) 0%, rgba(18,18,18,0.96) 28%, #121212 100%);
      flex-shrink: 0;
      z-index: 20;
    }

    .composer-card {
      background-color: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 26px;
      padding: 14px 14px 10px 16px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.55);
      display: flex;
      flex-direction: column;
      gap: 10px;
      transition: border-color 0.2s ease;
    }

    .composer-card:focus-within {
      border-color: #3E3E44;
    }

    .composer-textarea {
      width: 100%;
      border: none;
      outline: none;
      background: transparent;
      color: var(--text-primary);
      font-family: inherit;
      font-size: 15.5px;
      line-height: 1.45;
      resize: none;
      max-height: 116px;
      min-height: 24px;
      overflow-y: auto;
    }

    .composer-textarea::placeholder {
      color: var(--text-muted);
    }

    .composer-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .toolbar-left, .toolbar-right {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .plus-btn {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      border: 1px solid var(--border-subtle);
      background-color: #262629;
      color: var(--text-secondary);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.16s ease;
    }

    .plus-btn:hover {
      color: var(--text-primary);
      background-color: #2E2E32;
    }

    .model-selector {
      border: none;
      background: transparent;
      color: var(--text-secondary);
      font-family: inherit;
      font-size: 13px;
      font-weight: 500;
      padding: 6px 10px;
      border-radius: 9999px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: color 0.16s ease, background-color 0.16s ease;
    }

    .model-selector:hover {
      color: var(--text-primary);
      background-color: rgba(255, 255, 255, 0.05);
    }

    .mic-btn {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      border: none;
      background: transparent;
      color: var(--text-secondary);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.18s ease;
    }

    .mic-btn:hover {
      color: var(--text-primary);
      background-color: rgba(255, 255, 255, 0.06);
    }

    .mic-btn.recording {
      color: #FFFFFF;
      background-color: var(--accent-mic);
      box-shadow: 0 0 14px rgba(239, 68, 68, 0.65);
      animation: micPulse 1.2s infinite ease-in-out;
    }

    @keyframes micPulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.08); }
    }

    .primary-action-btn {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      border: none;
      background-color: #FFFFFF;
      color: #121212;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: transform 0.16s ease, opacity 0.16s ease;
    }

    .primary-action-btn:hover {
      transform: scale(1.04);
    }

    .primary-action-btn:active {
      transform: scale(0.94);
    }

    /* Slide-over Drawer */
    .drawer-backdrop {
      position: absolute;
      inset: 0;
      background: rgba(0, 0, 0, 0.6);
      backdrop-filter: blur(4px);
      z-index: 40;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.22s ease;
    }

    .drawer-backdrop.open {
      opacity: 1;
      pointer-events: auto;
    }

    .drawer-panel {
      position: absolute;
      top: 0;
      left: 0;
      bottom: 0;
      width: 280px;
      background-color: #18181B;
      border-right: 1px solid var(--border-subtle);
      z-index: 50;
      padding: 20px 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      transform: translateX(-100%);
      transition: transform 0.25s cubic-bezier(0.22, 1, 0.36, 1);
    }

    .drawer-panel.open {
      transform: translateX(0);
    }

    .drawer-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border-subtle);
    }

    .drawer-title {
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--text-secondary);
    }

    .drawer-btn {
      width: 100%;
      padding: 11px 14px;
      border-radius: 12px;
      border: 1px solid var(--border-subtle);
      background-color: #222226;
      color: var(--text-primary);
      font-size: 14px;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 10px;
      cursor: pointer;
      transition: background-color 0.16s ease;
    }

    .drawer-btn:hover {
      background-color: #2A2A2F;
    }

    /* Quick Directives Popover (+) */
    .quick-menu {
      position: absolute;
      left: 16px;
      bottom: 118px;
      background-color: #1F1F23;
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      padding: 6px;
      width: 235px;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.65);
      z-index: 30;
      display: none;
      flex-direction: column;
      gap: 2px;
    }

    .quick-menu.open {
      display: flex;
    }

    .quick-item {
      border: none;
      background: transparent;
      color: var(--text-primary);
      text-align: left;
      padding: 9px 12px;
      border-radius: 10px;
      font-size: 13px;
      cursor: pointer;
      transition: background-color 0.15s ease;
    }

    .quick-item:hover {
      background-color: rgba(255, 255, 255, 0.07);
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <!-- Slide-over Hamburger Drawer -->
    <div id="drawerBackdrop" class="drawer-backdrop" onclick="toggleDrawer(false)"></div>
    <aside id="drawerPanel" class="drawer-panel">
      <div class="drawer-header">
        <span class="drawer-title">JARVIS OS · v2.5</span>
        <button class="icon-btn" onclick="toggleDrawer(false)" aria-label="Close menu">✕</button>
      </div>
      <button class="drawer-btn" onclick="resetConversation()">
        <span>＋</span>
        <span>New Session</span>
      </button>
      <button class="drawer-btn" id="ttsToggleBtn" onclick="toggleAutoSpeak()">
        <span id="ttsToggleIcon">🔊</span>
        <span id="ttsToggleLabel">Voice Readback: On</span>
      </button>
      <div style="margin-top:auto; font-size:12px; color:var(--text-muted); line-height:1.5;">
        Powered by Google GenAI SDK<br/>Model: <code>gemini-2.5-flash</code>
      </div>
    </aside>

    <!-- Top Header -->
    <header class="top-header">
      <button class="icon-btn" onclick="toggleDrawer(true)" aria-label="Menu">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round">
          <line x1="4" y1="7" x2="20" y2="7"></line>
          <line x1="4" y1="12" x2="16" y2="12"></line>
          <line x1="4" y1="17" x2="20" y2="17"></line>
        </svg>
      </button>

      <div class="status-pill" id="statusPill">
        <span class="status-dot" id="statusDot"></span>
        <span id="statusText">JARVIS · Online</span>
      </div>

      <button class="icon-btn" id="incognitoBtn" onclick="toggleIncognito()" aria-label="Toggle Incognito Chat" title="Ghost / Incognito Mode">
        <!-- Ghost / Incognito SVG -->
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M9 10h.01"></path>
          <path d="M15 10h.01"></path>
          <path d="M12 2a8 8 0 0 0-8 8v12l3-3 2.5 2.5L12 19l2.5 2.5L17 19l3 3V10a8 8 0 0 0-8-8z"></path>
        </svg>
      </button>
    </header>

    <!-- Main Viewport -->
    <main class="main-viewport">
      <!-- Empty Home State (Left Phone Screen) -->
      <section id="emptyState" class="empty-state">
        <svg class="jarvis-emblem" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="32 4 56 18 56 46 32 60 8 46 8 18 32 4"></polygon>
          <polygon points="32 14 47 23 47 41 32 50 17 41 17 23 32 14"></polygon>
          <line x1="32" y1="4" x2="32" y2="14"></line>
          <line x1="56" y1="18" x2="47" y2="23"></line>
          <line x1="56" y1="46" x2="47" y2="41"></line>
          <line x1="32" y1="60" x2="32" y2="50"></line>
          <line x1="8" y1="46" x2="17" y2="41"></line>
          <line x1="8" y1="18" x2="17" y2="23"></line>
          <circle cx="32" cy="32" r="4"></circle>
        </svg>
        <h1 id="greetingHeading" class="serif-greeting">Good Evening, Boss</h1>
        <p id="incognitoCaption" class="incognito-caption">Ghost Protocol active · History won't be saved</p>
      </section>

      <!-- Active Chat View (Right Phone Screen) -->
      <section id="chatFeed" class="chat-feed hidden" aria-live="polite"></section>

      <!-- Quick Directives Menu (+) -->
      <div id="quickMenu" class="quick-menu">
        <button class="quick-item" onclick="useQuickPrompt('Give me a concise executive briefing for today.')">⚡ Executive Briefing</button>
        <button class="quick-item" onclick="useQuickPrompt('Run a quick systems diagnostic and status report.')">🛰️ Systems Diagnostic</button>
        <button class="quick-item" onclick="useQuickPrompt('Draft a sharp, diplomatic reply to a high-priority client.')">✉️ Draft Executive Reply</button>
        <button class="quick-item" onclick="useQuickPrompt('Explain quantum error correction in 3 crisp bullet points.')">🧠 Rapid Technical Brief</button>
      </div>
    </main>

    <!-- Bottom Floating Input Card -->
    <footer class="composer-wrapper">
      <div class="composer-card">
        <textarea
          id="messageInput"
          class="composer-textarea"
          rows="1"
          placeholder="How can I help you today?"
          aria-label="How can I help you today?"
        ></textarea>
        <div class="composer-toolbar">
          <div class="toolbar-left">
            <button type="button" class="plus-btn" onclick="toggleQuickMenu()" aria-label="Quick directives">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
                <line x1="12" y1="5" x2="12" y2="19"></line>
                <line x1="5" y1="12" x2="19" y2="12"></line>
              </svg>
            </button>
            <button type="button" class="model-selector" id="modelSelectorBtn" onclick="cycleModel()">
              <span id="modelLabel">JARVIS 2.5</span>
              <span style="font-size: 11px; opacity: 0.75;">⌄</span>
            </button>
          </div>

          <div class="toolbar-right">
            <button type="button" class="mic-btn" id="micBtn" onclick="toggleVoiceInput()" aria-label="Voice input">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="22"></line>
              </svg>
            </button>

            <button type="button" class="primary-action-btn" id="primaryActionBtn" onclick="handlePrimaryAction()" aria-label="Send or Voice">
              <!-- Waveform icon shown when input is empty -->
              <svg id="waveformIcon" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round">
                <line x1="6" y1="9" x2="6" y2="15"></line>
                <line x1="10" y1="5" x2="10" y2="19"></line>
                <line x1="14" y1="8" x2="14" y2="16"></line>
                <line x1="18" y1="10" x2="18" y2="14"></line>
              </svg>
              <!-- Send upward arrow icon shown when input has text -->
              <svg id="sendArrowIcon" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" style="display:none;">
                <line x1="12" y1="19" x2="12" y2="5"></line>
                <polyline points="5 12 12 5 19 12"></polyline>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </footer>
  </div>

  <script>
    const emptyState = document.getElementById('emptyState');
    const chatFeed = document.getElementById('chatFeed');
    const greetingHeading = document.getElementById('greetingHeading');
    const messageInput = document.getElementById('messageInput');
    const waveformIcon = document.getElementById('waveformIcon');
    const sendArrowIcon = document.getElementById('sendArrowIcon');
    const micBtn = document.getElementById('micBtn');
    const statusText = document.getElementById('statusText');
    const statusDot = document.getElementById('statusDot');
    const incognitoBtn = document.getElementById('incognitoBtn');
    const incognitoCaption = document.getElementById('incognitoCaption');
    const quickMenu = document.getElementById('quickMenu');

    let isIncognito = false;
    let autoSpeak = false;
    let isRecording = false;
    let isLoading = false;
    let recognition = null;
    let incognitoHistory = [];

    // 1. Dynamic Greeting based on time of day
    function updateTimeGreeting() {
      const hour = new Date().getHours();
      let greeting = 'Good Evening, Boss';
      if (hour >= 5 && hour < 12) {
        greeting = 'Good Morning, Boss';
      } else if (hour >= 12 && hour < 17) {
        greeting = 'Good Afternoon, Boss';
      }
      greetingHeading.textContent = greeting;
    }
    updateTimeGreeting();

    // 2. Auto-resize textarea & toggle Waveform vs Send icon
    function syncComposerState() {
      messageInput.style.height = 'auto';
      messageInput.style.height = Math.min(messageInput.scrollHeight, 116) + 'px';
      const hasText = messageInput.value.trim().length > 0;
      waveformIcon.style.display = hasText ? 'none' : 'block';
      sendArrowIcon.style.display = hasText ? 'block' : 'none';
    }

    messageInput.addEventListener('input', syncComposerState);
    messageInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (messageInput.value.trim().length > 0) {
          sendMessage();
        }
      }
    });

    // 3. Markdown-lite formatter for clean JARVIS plain-text replies
    function formatJarvisText(raw) {
      const escaped = raw
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      const withCodeBlocks = escaped.replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>');
      const withInlineCode = withCodeBlocks.replace(/`([^`]+)`/g, '<code>$1</code>');
      const withBold = withInlineCode.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
      return withBold
        .split(/\\n\\n+/)
        .map(p => `<p>${p.replace(/\\n/g, '<br/>')}</p>`)
        .join('');
    }

    // 4. Render messages into Chat Feed
    function appendMessageToDOM(role, text) {
      emptyState.classList.add('hidden');
      chatFeed.classList.remove('hidden');

      const row = document.createElement('div');
      row.className = `message-row ${role}`;

      if (role === 'user') {
        const bubble = document.createElement('div');
        bubble.className = 'user-bubble';
        bubble.textContent = text;
        row.appendChild(bubble);
      } else {
        const body = document.createElement('div');
        body.className = 'jarvis-response';
        body.innerHTML = formatJarvisText(text);
        row.appendChild(body);

        const actions = document.createElement('div');
        actions.className = 'message-actions';

        // Copy button
        const copyBtn = document.createElement('button');
        copyBtn.className = 'action-icon-btn';
        copyBtn.title = 'Copy response';
        copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
        copyBtn.onclick = () => {
          navigator.clipboard.writeText(text);
          copyBtn.style.color = '#10B981';
          setTimeout(() => { copyBtn.style.color = ''; }, 1200);
        };

        // Regenerate button
        const regenBtn = document.createElement('button');
        regenBtn.className = 'action-icon-btn';
        regenBtn.title = 'Regenerate response';
        regenBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2v6h-6"></path><path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path><path d="M3 22v-6h6"></path><path d="M21 12a9 9 0 0 1-15 6.7L3 16"></path></svg>`;
        regenBtn.onclick = () => regenerateLastResponse();

        // Speak button
        const speakBtn = document.createElement('button');
        speakBtn.className = 'action-icon-btn';
        speakBtn.title = 'Read aloud';
        speakBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>`;
        speakBtn.onclick = () => speakText(text);

        actions.appendChild(copyBtn);
        actions.appendChild(regenBtn);
        actions.appendChild(speakBtn);
        row.appendChild(actions);
      }

      chatFeed.appendChild(row);
      chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function showTypingIndicator() {
      const row = document.createElement('div');
      row.id = 'typingRow';
      row.className = 'message-row model';
      row.innerHTML = `<div class="typing-indicator"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>`;
      chatFeed.appendChild(row);
      chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function removeTypingIndicator() {
      const el = document.getElementById('typingRow');
      if (el) el.remove();
    }

    // 5. Send Message & Regenerate API calls
    async function sendMessage(overrideText, viaVoice = false) {
      const text = (overrideText !== undefined ? overrideText : messageInput.value).trim();
      if (!text || isLoading) return;

      quickMenu.classList.remove('open');
      messageInput.value = '';
      syncComposerState();
      appendMessageToDOM('user', text);

      isLoading = true;
      showTypingIndicator();

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            incognito: isIncognito,
            history: isIncognito ? incognitoHistory : undefined
          })
        });
        const data = await res.json();
        removeTypingIndicator();

        const reply = data.reply || 'Telemetry interrupted, Boss.';
        if (isIncognito) {
          incognitoHistory.push({ role: 'user', text });
          incognitoHistory.push({ role: 'model', text: reply });
        }
        appendMessageToDOM('model', reply);

        if (autoSpeak || viaVoice) {
          speakText(reply);
        }
      } catch (err) {
        removeTypingIndicator();
        appendMessageToDOM('model', 'Network link interrupted, Boss. Please verify the local connection.');
      } finally {
        isLoading = false;
      }
    }

    async function regenerateLastResponse() {
      if (isLoading) return;
      const modelRows = chatFeed.querySelectorAll('.message-row.model');
      if (modelRows.length > 0) {
        modelRows[modelRows.length - 1].remove();
      }
      isLoading = true;
      showTypingIndicator();

      try {
        const res = await fetch('/api/regenerate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            incognito: isIncognito,
            history: isIncognito ? incognitoHistory : undefined
          })
        });
        const data = await res.json();
        removeTypingIndicator();
        const reply = data.reply || 'Unable to regenerate response, Boss.';
        if (isIncognito && incognitoHistory.length > 0) {
          incognitoHistory[incognitoHistory.length - 1] = { role: 'model', text: reply };
        }
        appendMessageToDOM('model', reply);
        if (autoSpeak) speakText(reply);
      } catch (e) {
        removeTypingIndicator();
        appendMessageToDOM('model', 'Regeneration failed, Boss.');
      } finally {
        isLoading = false;
      }
    }

    // 6. Web Speech API (SpeechRecognition & speechSynthesis)
    function initSpeechRecognition() {
      const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRec) return null;
      const rec = new SpeechRec();
      rec.continuous = false;
      rec.interimResults = true;
      rec.lang = 'en-US';

      rec.onstart = () => {
        isRecording = true;
        micBtn.classList.add('recording');
        statusText.textContent = 'JARVIS · Listening...';
      };

      rec.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        messageInput.value = transcript;
        syncComposerState();
      };

      rec.onend = () => {
        const captured = messageInput.value.trim();
        isRecording = false;
        micBtn.classList.remove('recording');
        statusText.textContent = isIncognito ? 'JARVIS · Incognito' : 'JARVIS · Online';
        if (captured) {
          sendMessage(captured, true);
        }
      };

      rec.onerror = () => {
        isRecording = false;
        micBtn.classList.remove('recording');
        statusText.textContent = isIncognito ? 'JARVIS · Incognito' : 'JARVIS · Online';
      };

      return rec;
    }

    function toggleVoiceInput() {
      if (!recognition) {
        recognition = initSpeechRecognition();
      }
      if (!recognition) {
        alert('SpeechRecognition is not supported in this browser. Try Chrome, Edge, or Safari.');
        return;
      }
      if (isRecording) {
        recognition.stop();
      } else {
        window.speechSynthesis.cancel();
        recognition.start();
      }
    }

    function speakText(text) {
      if (!('speechSynthesis' in window)) return;
      window.speechSynthesis.cancel();
      const cleanText = text.replace(/[*#`_~]/g, '');
      const utterance = new SpeechSynthesisUtterance(cleanText);
      const voices = window.speechSynthesis.getVoices();
      const preferred = voices.find(v => /Daniel|Arthur|Google UK English Male|en-GB/i.test(v.name || v.lang));
      if (preferred) utterance.voice = preferred;
      utterance.rate = 1.03;
      utterance.pitch = 0.96;
      window.speechSynthesis.speak(utterance);
    }

    function handlePrimaryAction() {
      if (messageInput.value.trim().length > 0) {
        sendMessage();
      } else {
        toggleVoiceInput();
      }
    }

    // 7. UI Controls (Drawer, Incognito, Quick Menu)
    function toggleDrawer(open) {
      document.getElementById('drawerBackdrop').classList.toggle('open', open);
      document.getElementById('drawerPanel').classList.toggle('open', open);
    }

    function toggleIncognito() {
      isIncognito = !isIncognito;
      incognitoBtn.classList.toggle('active-incognito', isIncognito);
      statusDot.classList.toggle('incognito', isIncognito);
      statusText.textContent = isIncognito ? 'JARVIS · Incognito' : 'JARVIS · Online';
      incognitoCaption.classList.toggle('visible', isIncognito);
    }

    function toggleAutoSpeak() {
      autoSpeak = !autoSpeak;
      document.getElementById('ttsToggleIcon').textContent = autoSpeak ? '🔊' : '🔇';
      document.getElementById('ttsToggleLabel').textContent = `Voice Readback: ${autoSpeak ? 'On' : 'Off'}`;
    }

    function toggleQuickMenu() {
      quickMenu.classList.toggle('open');
    }

    function useQuickPrompt(promptText) {
      quickMenu.classList.remove('open');
      sendMessage(promptText);
    }

    function cycleModel() {
      const label = document.getElementById('modelLabel');
      label.textContent = label.textContent === 'JARVIS 2.5' ? 'JARVIS 2.5 Flash' : 'JARVIS 2.5';
    }

    async function resetConversation() {
      toggleDrawer(false);
      incognitoHistory = [];
      chatFeed.innerHTML = '';
      chatFeed.classList.add('hidden');
      emptyState.classList.remove('hidden');
      updateTimeGreeting();
      await fetch('/api/reset', { method: 'POST' });
    }

    // Load existing session history on page open
    (async function loadExistingSession() {
      try {
        const res = await fetch('/api/history');
        const data = await res.json();
        if (Array.isArray(data.history) && data.history.length > 0) {
          data.history.forEach(item => appendMessageToDOM(item.role, item.text));
        }
      } catch (e) {}
    })();
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    get_session_id()
    return render_template_string(HTML_TEMPLATE)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY"))})


@app.route("/api/history", methods=["GET"])
def get_history():
    sid = get_session_id()
    return jsonify({"history": CONVERSATION_STORE.get(sid, [])})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    incognito = bool(data.get("incognito", False))

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    if incognito:
        client_history = data.get("history") or []
        temp_history = [
            {"role": item.get("role", "user"), "text": item.get("text", "")}
            for item in client_history
            if item.get("text")
        ]
        temp_history.append({"role": "user", "text": user_message})
        try:
            reply = generate_jarvis_reply(temp_history)
        except Exception as exc:
            reply = f"Apologies, Boss. Transient telemetry fault: {exc}"
        return jsonify({"reply": reply, "incognito": True})

    sid = get_session_id()
    history = CONVERSATION_STORE.setdefault(sid, [])
    history.append({"role": "user", "text": user_message})

    try:
        reply = generate_jarvis_reply(history)
    except Exception as exc:
        reply = f"Apologies, Boss. I encountered a transient telemetry fault: {exc}"

    history.append({"role": "model", "text": reply})
    return jsonify({"reply": reply, "history_length": len(history)})


@app.route("/api/regenerate", methods=["POST"])
def regenerate():
    data = request.get_json(silent=True) or {}
    incognito = bool(data.get("incognito", False))

    if incognito:
        client_history = data.get("history") or []
        if client_history and client_history[-1].get("role") == "model":
            client_history = client_history[:-1]
        if not client_history:
            return jsonify({"reply": "No previous directive to regenerate, Boss."})
        try:
            reply = generate_jarvis_reply(client_history)
        except Exception as exc:
            reply = f"Apologies, Boss. Regeneration error: {exc}"
        return jsonify({"reply": reply})

    sid = get_session_id()
    history = CONVERSATION_STORE.get(sid, [])
    if history and history[-1]["role"] == "model":
        history.pop()

    if not history:
        return jsonify({"reply": "No previous directive to regenerate, Boss."})

    try:
        reply = generate_jarvis_reply(history)
    except Exception as exc:
        reply = f"Apologies, Boss. Regeneration encountered an error: {exc}"

    history.append({"role": "model", "text": reply})
    return jsonify({"reply": reply})


@app.route("/api/reset", methods=["POST"])
def reset():
    sid = get_session_id()
    CONVERSATION_STORE[sid] = []
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
