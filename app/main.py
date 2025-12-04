from pathlib import Path
from typing import List, Dict, Tuple, Any

from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from dotenv import load_dotenv
import os
import re
import uuid
from datetime import datetime
import json

import google.generativeai as genai

try:
    from supabase import create_client  # type: ignore
except ImportError:  # ライブラリ未インストール時でもアプリが落ちないようにする
    create_client = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "templates"

# 状態保存用のパスは環境変数で上書き可能にしておく（ファイル保存のフォールバック用）
custom_state_file = os.getenv("BRAINSTORM_STATE_FILE")
if custom_state_file:
    STATE_FILE = Path(custom_state_file)
    DATA_DIR = STATE_FILE.parent
else:
    DATA_DIR = BASE_DIR / "data"
    STATE_FILE = DATA_DIR / "state.json"

load_dotenv()

# Gemini API の設定（APIキーは環境変数 GEMINI_API_KEY から読む）
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Supabase クライアント設定（設定されていない場合は None のまま）
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_TABLE = "app_state"
SUPABASE_STATE_ID = "default"
supabase_client: Any | None = None
if create_client and SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception:
        supabase_client = None

# デバッグログの出力有無（プロンプトやレスポンスをログに出すかどうか）
DEBUG_MODE = os.getenv("BRAINSTORM_DEBUG", "").lower() in {"1", "true", "yes"}

app = FastAPI(title="Mini Idobata Brainstorm")

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


class Message(BaseModel):
    id: str
    author: str
    content: str
    created_at: datetime


class Idea(BaseModel):
    id: str
    session_id: str
    author: str
    title: str
    description: str
    created_at: datetime


class PersonalMessage(BaseModel):
    id: str
    session_id: str
    user_id: str
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime


class Session(BaseModel):
    id: str
    title: str
    description: str
    created_at: datetime
    messages: List[Message] = []
    last_summary: str | None = None


# 簡易インメモリDB（本番運用する場合はDBに置き換え）
SESSIONS: Dict[str, Session] = {}
PERSONAL_CHATS: Dict[Tuple[str, str], List[PersonalMessage]] = {}
IDEAS: List[Idea] = []


def _ensure_data_dir() -> None:
    """データ保存用ディレクトリを作成しておく。"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # ディレクトリ作成に失敗してもアプリ自体は動くようにする
        pass


def _save_state() -> None:
    """
    現在のセッション情報を JSON に永続化する。
    少人数利用を想定した簡易実装なので、毎回全体を書き出している。
    """
    # まずは Supabase に保存を試みる（設定されている場合）
    data: Dict[str, Any] = {
        "sessions": [s.dict() for s in SESSIONS.values()],
        "personal_chats": [
            {
                "session_id": sid,
                "user_id": uid,
                "history": [m.dict() for m in history],
            }
            for (sid, uid), history in PERSONAL_CHATS.items()
        ],
        "ideas": [i.dict() for i in IDEAS],
    }

    if supabase_client:
        try:
            supabase_client.table(SUPABASE_TABLE).upsert(
                {"id": SUPABASE_STATE_ID, "data": data}
            ).execute()
            return
        except Exception:
            # Supabase 保存に失敗した場合は、ローカルファイルへの保存にフォールバックする
            pass

    # Supabase が無い・失敗した場合はローカルファイルに保存
    _ensure_data_dir()
    try:
        STATE_FILE.write_text(
            json.dumps(data, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # 保存に失敗してもチャット自体は続けられるように、ここでは例外を握りつぶす
        pass


def _load_state() -> None:
    """
    起動時に永続化されたセッション情報を読み込む。
    Supabase が設定されていればそちらを優先し、なければローカルの JSON ファイルを読む。
    """
    data: Dict[str, Any] | None = None

    # Supabase からの読み込みを優先
    if supabase_client:
        try:
            res = (
                supabase_client.table(SUPABASE_TABLE)
                .select("data")
                .eq("id", SUPABASE_STATE_ID)
                .execute()
            )
            rows = getattr(res, "data", []) or []
            if rows:
                data = rows[0].get("data") or {}
        except Exception:
            data = None

    # Supabase で取得できなかった場合はローカルファイルから読み込む
    if data is None:
        if not STATE_FILE.exists():
            return
        try:
            raw = STATE_FILE.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:
            return

    sessions_raw = data.get("sessions") or []
    personal_chats_raw = data.get("personal_chats") or []
    ideas_raw = data.get("ideas") or []

    SESSIONS.clear()
    PERSONAL_CHATS.clear()
    IDEAS.clear()

    # Session とその中の Message
    for s in sessions_raw:
        try:
            session = Session(**s)
            SESSIONS[session.id] = session
        except Exception:
            continue

    # 個人チャット履歴
    for pc in personal_chats_raw:
        try:
            sid = pc["session_id"]
            uid = pc["user_id"]
            history = [PersonalMessage(**m) for m in pc.get("history", [])]
            PERSONAL_CHATS[(sid, uid)] = history
        except Exception:
            continue

    # アイデア
    for i in ideas_raw:
        try:
            idea = Idea(**i)
            IDEAS.append(idea)
        except Exception:
            continue


# アプリ起動時に一度だけ状態を読み込む
_load_state()


def _extract_text_from_gemini_response(response: Any) -> str | None:
    """
    google-generativeai のレスポンス構造の違いにある程度耐性を持たせて、
    できるだけテキストを取り出すためのヘルパー。
    """
    # 0. 素の文字列として返ってきた場合
    if isinstance(response, str) and response.strip():
        return response.strip()

    # 1. まずは response.text を優先
    try:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass

    # 2. candidates -> content.parts[*].text を見る
    try:
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            fragments: list[str] = []
            for p in parts:
                # Part オブジェクトの text 属性
                t = getattr(p, "text", None)
                if isinstance(t, str) and t.strip():
                    fragments.append(t)
                    continue
                # dict として返ってきた場合
                if isinstance(p, dict):
                    t2 = p.get("text")
                    if isinstance(t2, str) and t2.strip():
                        fragments.append(t2)
            text_joined = "\n".join(fragments).strip()
            if text_joined:
                return text_joined
    except Exception:
        pass

    # 3. 念のため content 自体が str の場合
    try:
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
    except Exception:
        pass

    # 4. dict / list / 任意オブジェクトを再帰的に走査して "text" キーを探す
    def _find_text_anywhere(obj: Any, depth: int = 0) -> str | None:
        if depth > 4 or obj is None:
            return None

        # 文字列そのもの
        if isinstance(obj, str):
            if obj.strip():
                return obj.strip()
            return None

        # dict の場合は "text" というキーを優先して見る
        if isinstance(obj, dict):
            text_val = obj.get("text")
            if isinstance(text_val, str) and text_val.strip():
                return text_val.strip()
            for v in obj.values():
                found = _find_text_anywhere(v, depth + 1)
                if found:
                    return found
            return None

        # list / tuple の場合は順に探索
        if isinstance(obj, (list, tuple)):
            for v in obj:
                found = _find_text_anywhere(v, depth + 1)
                if found:
                    return found
            return None

        # 任意オブジェクトは __dict__ を辿る
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            return _find_text_anywhere(d, depth + 1)

        return None

    try:
        generic_text = _find_text_anywhere(response)
        if generic_text:
            return generic_text
    except Exception:
        pass

    return None


def generate_local_facilitator_reply(latest_user_message: str) -> str:
    """
    Gemini からうまく応答を取得できなかった場合のローカル fallback。
    きわめてシンプルだが、少なくともエラーメッセージよりは会話として自然な文を返す。
    """
    latest = latest_user_message.strip()
    if not latest:
        latest = "いまのお気持ちや状況"

    first = f"{latest}と感じていらっしゃるのですね。"
    second = (
        "その中で、特に大事にしたいポイントはどんなところでしょうか？"
        "たとえば「こうなっていたら理想的だな」と思う状態があれば、言葉にしてみてもらえますか。"
    )
    third = (
        "また、普段の過ごし方や制約との違いで気になっている点があれば、"
        "具体的なエピソードを1つ挙げて教えていただけると、さらに整理しやすくなりそうです。"
    )
    return "\n".join([first, second, third])


def generate_gemini_text(
    prompt: str,
    temperature: float = 0.6,
    max_retries: int = 2,
    model_override: str | None = None,
) -> str:
    """
    Gemini API を使ってテキストを生成するヘルパー関数。
    モデル名は環境変数 BRAINSTORM_MODEL で上書き可能（なければ 2.0 flash を使う）。
    """
    if not gemini_api_key:
        return "Gemini APIキー(GEMINI_API_KEY)が設定されていないため、AI応答を生成できません。"

    # デフォルトは汎用性の高い 2.0 flash モデル。
    # model_override > 環境変数 BRAINSTORM_MODEL > デフォルト の優先順位で決める。
    model_name = (
        model_override
        or os.getenv("BRAINSTORM_MODEL")
        or "models/gemini-2.0-flash"
    )
    # list_models の結果に合わせて、先頭に models/ が無ければ付ける
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    # 同じプロンプトで最大 max_retries 回までリトライする
    last_response: Any | None = None

    try:
        model = genai.GenerativeModel(model_name)
        for attempt in range(max_retries + 1):
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    # アイデア抽出や短い応答に十分な長さ
                    max_output_tokens=256,
                ),
            )
            last_response = response

            # デバッグ用: レスポンスの概要をログに出す（長すぎないように先頭だけ）
            if DEBUG_MODE:
                try:
                    print(
                        f"=== DEBUG generate_gemini_text attempt {attempt + 1}/"
                        f"{max_retries + 1}: raw response repr (truncated) ==="
                    )
                    print("response:")
                    print(repr(response)[:800])
                    print("=== DEBUG END ===")
                except Exception:
                    pass

            # レスポンスから可能な限りテキストを取り出す
            text = _extract_text_from_gemini_response(response)
            if text:
                return text

        # ここまで来たら、全ての試行でテキストが取得できなかった
        response = last_response

        # セーフティフィルタでブロックされた場合のメッセージ
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason and str(block_reason) != "BLOCK_REASON_UNSPECIFIED":
            return (
                "安全フィルタにより回答がブロックされました。"
                "表現を少し穏やかにするなど、別の聞き方でもう一度試してみてください。"
            )

        # 想定外のレスポンス形式
        return (
            "AIからの応答をうまく読み取れませんでした。"
            "少し時間をおいてから、または別の聞き方でもう一度試してみてください。"
        )

    except Exception as e:
        return f"Gemini呼び出しでエラーが発生しました: {e}"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    sessions = sorted(SESSIONS.values(), key=lambda s: s.created_at, reverse=True)

    # 各セッションごとの「メインに出されたアイデア」の件数を集計
    idea_counts: Dict[str, int] = {}
    for idea in IDEAS:
        idea_counts[idea.session_id] = idea_counts.get(idea.session_id, 0) + 1

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "sessions": sessions,
            "idea_counts": idea_counts,
        },
    )


@app.post("/sessions", response_class=RedirectResponse)
async def create_session(
    title: str = Form(...),
    description: str = Form(""),
):
    session_id = str(uuid.uuid4())
    session = Session(
        id=session_id,
        title=title,
        description=description,
        created_at=datetime.now(),
        messages=[],
        last_summary=None,
    )
    SESSIONS[session_id] = session
    _save_state()
    return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)


@app.get("/sessions/{session_id}", response_class=HTMLResponse)
async def get_session(
    request: Request,
    session_id: str,
):
    session = SESSIONS.get(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)

    session_ideas = [idea for idea in IDEAS if idea.session_id == session_id]

    return templates.TemplateResponse(
        "session.html",
        {
            "request": request,
            "session": session,
            "ideas": sorted(
                session_ideas, key=lambda i: i.created_at, reverse=True
            ),
        },
    )


@app.post("/sessions/{session_id}/delete", response_class=RedirectResponse)
async def delete_session(session_id: str):
    """
    セッション本体と、関連する個人チャット履歴・アイデアをすべて削除する。
    削除後はトップページにリダイレクトする。
    """
    session = SESSIONS.pop(session_id, None)
    if session is None:
        # 既に存在しない場合も、単にトップに戻す
        return RedirectResponse(url="/", status_code=303)

    # 個人チャット履歴を削除
    keys_to_delete = [key for key in PERSONAL_CHATS.keys() if key[0] == session_id]
    for key in keys_to_delete:
        PERSONAL_CHATS.pop(key, None)

    # このセッションに紐づくアイデアを削除（リストをインプレースで更新）
    IDEAS[:] = [idea for idea in IDEAS if idea.session_id != session_id]

    _save_state()
    return RedirectResponse(url="/", status_code=303)


@app.post(
    "/sessions/{session_id}/ideas/{idea_id}/delete",
    response_class=RedirectResponse,
)
async def delete_idea(session_id: str, idea_id: str):
    """
    メインに出されたアイデア（Idea）を1件削除する。
    セッション自体は残しつつ、そのアイデアだけを消す。
    """
    # セッションが存在しない場合は一覧へ
    session = SESSIONS.get(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)

    # ID が一致するアイデアだけ取り除く
    before = len(IDEAS)
    IDEAS[:] = [i for i in IDEAS if i.id != idea_id]
    after = len(IDEAS)
    if before != after:
        _save_state()

    return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)


@app.post("/sessions/{session_id}/messages", response_class=RedirectResponse)
async def add_message(
    session_id: str,
    author: str = Form("匿名"),
    content: str = Form(...),
):
    session = SESSIONS.get(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)

    if content.strip():
        msg = Message(
            id=str(uuid.uuid4()),
            author=author or "匿名",
            content=content.strip(),
            created_at=datetime.now(),
        )
        session.messages.append(msg)

    _save_state()
    return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)


@app.post(
    "/sessions/{session_id}/personal/enter",
    response_class=RedirectResponse,
)
async def enter_personal_chat(
    session_id: str,
    user_name: str = Form(...),
):
    """
    セッション詳細ページ上のフォームから、ユーザー名を入力して
    個人チャットページへリダイレクトするエンドポイント。
    """
    session = SESSIONS.get(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)

    name = user_name.strip() or "匿名"
    return RedirectResponse(
        url=f"/sessions/{session_id}/personal?user={name}",
        status_code=303,
    )

@app.get("/sessions/{session_id}/personal", response_class=HTMLResponse)
async def personal_chat(
    request: Request,
    session_id: str,
    user: str = Query(..., alias="user"),
):
    """
    個人用のAIチャット画面。
    user はブラウザごとに適当な名前（ニックネーム）を指定してアクセスする想定。
    """
    session = SESSIONS.get(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)

    key = (session_id, user)
    history = PERSONAL_CHATS.get(key, [])

    return templates.TemplateResponse(
        "personal_chat.html",
        {
            "request": request,
            "session": session,
            "user_id": user,
            "history": history,
        },
    )


@app.post("/sessions/{session_id}/personal/messages", response_class=RedirectResponse)
async def personal_chat_message(
    session_id: str,
    user_id: str = Form(...),
    content: str = Form(...),
):
    """
    個人チャットでのユーザー発言を受け取り、ローカルLLMに投げてAI応答を追加する。
    """
    session = SESSIONS.get(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)

    text = content.strip()
    if not text:
        return RedirectResponse(
            url=f"/sessions/{session_id}/personal?user={user_id}",
            status_code=303,
        )

    key = (session_id, user_id)
    history = PERSONAL_CHATS.setdefault(key, [])

    # ユーザー発言を履歴に追加
    user_msg = PersonalMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=text,
        created_at=datetime.now(),
    )
    history.append(user_msg)

    # これまでの個人チャット履歴すべてをテキストとしてまとめる
    conversation_text = "\n".join(
        [f"{m.role}: {m.content}" for m in history]
    )

    # セッション情報と要約を前提コンテキストとして渡しつつ、
    # 出力ルールは system_prompt にまとめる。
    session_description = session.description or "（説明なし）"
    summary_text = (
        session.last_summary
        or "まだ正式な要約はありません。これまでの会話とセッション情報から状況を推測してください。"
    )

    system_prompt = (
        "あなたは少人数チームのブレインストーミングを支援する、日本語のファシリテーターです。\n"
        "【役割と目的】このチャットのセッションタイトルと説明を手がかりに、ユーザーが考えているテーマの意味や目的を大切にしながら、"
        "最終的にそのテーマに対する1つ以上の具体的なアイデア（行動パターンや過ごし方、決め方など）が見えてくるように支援してください。"
        "AIが結論を一方的に決めつけず、途中の思考整理や深掘り、選択肢の拡張を一緒に行う「壁打ち」相手として寄り添ってください。\n"
        "【出力の考え方】\n"
        "・返信は日本語、です・ます調で、状況に応じて読みやすい長さ（目安として2〜4文）にしてください。\n"
        "・最初の1文目は「なるほど。」「いいですね。」「素敵です。」のようなごく短い一言の共感や相槌を。そのあとすぐに質問や提案、具体的な問いかけに進んでください。\n"
        "・直近2〜3ターンと同じ構成（共感→理由を尋ねる→選択肢を挙げる、など）にならないようにしてください。理由や背景を深掘りする質問／条件や制約を整理する質問／具体的な場面やエピソードを尋ねる質問／まったく別の角度からの問いかけ、など複数のパターンをローテーションするつもりで使ってください。\n"
        "・必要に応じて、ユーザーの条件や気持ちに合いそうな選択肢や視点をいくつか示し、「どれがしっくりきますか？」「他にどんなパターンがありそうですか？」のように、次の一歩を考えやすくしてください。\n"
        "・ときどき、それまでの会話から見えてきたアイデア候補や方向性を短く整理し、「今のところ〇〇という案が見えてきていますが、どう感じますか？」のようにユーザーと一緒に確かめてください。\n"
        "【禁止事項】\n"
        "・過去に自分がした質問とほぼ同じ内容を、言い換えて繰り返さないでください。\n"
        "・「詳しく教えてください」「もう少し具体的に教えてください」だけの汎用的な質問で終えないでください。\n"
        "・抽象的な一般論だけで終わらせず、必ずユーザーの具体的な発言や状況にひもづけて書いてください。\n"
        "・ユーザーの直前の発言を「◯◯ということですね」「◯◯ということでしょうか」「◯◯なのですね」のような形で長く言い換える一文要約は絶対に書かないでください（共感の一言だけで止めてください）。\n"
    )

    full_prompt = (
        "# 前提コンテキスト\n\n"
        "これまでの議論要約:\n"
        f"{summary_text}\n\n"
        "セッション情報:\n"
        f"- タイトル: 「{session.title}」\n"
        f"- 説明: 「{session_description}」\n\n"
        "# 会話履歴\n\n"
        "<history>\n"
        f"{conversation_text}\n"
        "</history>\n\n"
        "# 直近のユーザー発言\n\n"
        f"「{text}」\n\n"
        "# 指示\n\n"
        f"{system_prompt}\n"
        "上記の履歴と発言を踏まえ、出力ルールに従って回答してください。\n"
        f"特に、直近の発言にある「{text}」という意図に対し、ファシリテーターとして次の思考を促す問いかけを返してください。\n"
    )

    # デバッグ用ログ
    if DEBUG_MODE:
        print("=== DEBUG personal_chat_message ===")
        print("user_id:", user_id)
        print(
            "full_prompt (first 400 chars):",
            full_prompt[:400].replace("\n", "\\n"),
        )

    assistant_content = generate_gemini_text(full_prompt, temperature=0.8)

    if DEBUG_MODE:
        print(
            "assistant_content (repr, first 200 chars):",
            repr(assistant_content)[:200],
        )
        print("=== END DEBUG personal_chat_message ===")

    # AI応答を履歴に追加
    assistant_msg = PersonalMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=assistant_content,
        created_at=datetime.now(),
    )
    history.append(assistant_msg)

    _save_state()
    return RedirectResponse(
        url=f"/sessions/{session_id}/personal?user={user_id}",
        status_code=303,
    )


@app.post(
    "/sessions/{session_id}/personal/reset",
    response_class=RedirectResponse,
)
async def reset_personal_chat(
    session_id: str,
    user_id: str = Form(...),
):
    """
    個人チャットの履歴をリセットする（指定ユーザー分のみ）。
    セッション自体やメインのアイデアは残したまま、個人チャットだけを空にする。
    """
    session = SESSIONS.get(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)

    key = (session_id, user_id)
    if key in PERSONAL_CHATS:
        PERSONAL_CHATS.pop(key, None)
        _save_state()

    return RedirectResponse(
        url=f"/sessions/{session_id}/personal?user={user_id}",
        status_code=303,
    )


@app.post("/sessions/{session_id}/personal/promote", response_class=HTMLResponse)
async def promote_idea(
    request: Request,
    session_id: str,
    user_id: str = Form(...),
):
    """
    個人チャット履歴をもとに、Gemini に1つのアイデア案を生成してもらい、
    ユーザーがタイトルと本文を確認・修正できる画面を表示する。
    （この時点ではまだ IDEAS には保存しない）
    """
    session = SESSIONS.get(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)

    key = (session_id, user_id)
    history = PERSONAL_CHATS.get(key, [])
    if not history:
        return RedirectResponse(
            url=f"/sessions/{session_id}/personal?user={user_id}",
            status_code=303,
        )

    conversation_text = "\n".join(
        [f"{m.role}: {m.content}" for m in history]
    )

    session_description = session.description or "（説明なし）"
    system_prompt = (
        "# 命令\n"
        "以下の対話履歴から、セッション情報に最も合致する価値あるアイデアを1つ抽出し、出力形式に従って提示せよ。\n\n"
        "# セッション情報\n"
        f"- タイトル: {session.title}\n"
        f"- 説明: {session_description}\n\n"
        "# 出力形式\n"
        "- 1行目: タイトル（10〜20文字の日本語、印象的な短文）のみを書き、「タイトル:」「タイトル：」「- タイトル」などのラベルや記号は付けないこと。\n"
        "- 2行目以降: 本文（400文字以内の具体的内容）のみを書き、「本文:」「本文：」「- 本文」などのラベルや見出しは付けないこと。\n"
        "- 文体: 常体（だ・である調）\n"
    )

    user_prompt = f"対話履歴:\n{conversation_text}"

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    # アイデア出し用には、より軽量な flash-lite モデルを使用する
    raw_text = generate_gemini_text(
        full_prompt,
        temperature=0.5,
        model_override="models/gemini-2.0-flash-lite",
    )

    # 生成に失敗した場合の検知（generate_gemini_text のフォールバック文言）
    error_markers = [
        "AIからの応答をうまく読み取れませんでした。",
        "Gemini APIキー(GEMINI_API_KEY)が設定されていないため",
        "Gemini呼び出しでエラーが発生しました",
        "安全フィルタにより回答がブロックされました。",
    ]
    has_error = any(marker in raw_text for marker in error_markers)

    if has_error:
        # 自動生成に失敗した場合は、空欄＋エラーメッセージだけを渡す
        title = ""
        description = ""
        error_message = (
            "AIからアイデア案を自動生成できませんでした。"
            "下のフォームに、タイトルと内容を直接入力してください。"
        )
    else:
        lines = [line for line in raw_text.splitlines() if line.strip()]
        if not lines:
            title = ""
            description = ""
        else:
            # 1行目: タイトル候補
            raw_title = lines[0].strip()
            # 先頭に付いてしまいがちな「タイトル:」「- タイトル」などのラベルを除去
            title = re.sub(
                r"^(?:[-・]\s*)?(?:タイトル|題名)\s*[:：]\s*",
                "",
                raw_title,
            ).strip()

            # 2行目以降: 本文候補
            body_lines = lines[1:]
            if body_lines:
                # 先頭行に「本文:」「- 本文」などが付いている場合は削る
                body_lines[0] = re.sub(
                    r"^(?:[-・]\s*)?(?:本文|内容)\s*[:：]\s*",
                    "",
                    body_lines[0].strip(),
                )
            description = "\n".join(body_lines).strip() or title

        error_message = ""

    # ここではまだ保存せず、ユーザーに確認・編集してもらう
    return templates.TemplateResponse(
        "idea_confirm.html",
        {
            "request": request,
            "session": session,
            "user_id": user_id,
            "draft_title": title,
            "draft_description": description,
            "error_message": error_message,
        },
    )


@app.post(
    "/sessions/{session_id}/personal/promote/confirm",
    response_class=RedirectResponse,
)
async def confirm_promoted_idea(
    session_id: str,
    user_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
):
    """
    個人チャットから生成したアイデア案を、ユーザーが編集したうえで
    メインのアイデア一覧（IDEAS）に保存するエンドポイント。
    """
    session = SESSIONS.get(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)

    title = (title or "").strip()
    description = (description or "").strip()

    if not title and not description:
        # 何も内容がない場合は個人チャットに戻す
        return RedirectResponse(
            url=f"/sessions/{session_id}/personal?user={user_id}",
            status_code=303,
        )

    # タイトルと説明の長さを簡易に制限する（カード表示向け）
    max_title_chars = 20
    if len(title) > max_title_chars:
        title = title[:max_title_chars].rstrip() + "…"

    max_description_chars = 400
    if len(description) > max_description_chars:
        description = description[:max_description_chars].rstrip() + "…"

    idea = Idea(
        id=str(uuid.uuid4()),
        session_id=session_id,
        author=user_id,
        title=title or "アイデア（タイトル未設定）",
        description=description or title,
        created_at=datetime.now(),
    )
    IDEAS.append(idea)

    _save_state()
    return RedirectResponse(
        url=f"/sessions/{session_id}",
        status_code=303,
    )

@app.post("/sessions/{session_id}/summarize", response_class=RedirectResponse)
async def summarize_session(session_id: str):
    session = SESSIONS.get(session_id)
    if not session or not session.messages:
        return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)

    # すべてのメッセージを一つのテキストにまとめる
    messages_text = "\n".join(
        [f"- {m.author}: {m.content}" for m in session.messages]
    )

    session_description = session.description or "（説明なし）"
    system_prompt = (
        "あなたは少人数チームのブレインストーミングを支援するファシリテーターです。\n"
        "セッションのタイトルと説明は次の通りです。\n"
        f"- タイトル: 「{session.title}」\n"
        f"- 説明: 「{session_description}」\n"
        "以下の発言一覧を読み、このセッションの目的に沿う形で論点ごとに整理し、日本語でわかりやすく要約してください。\n"
        "出力フォーマットは次のようにしてください：\n\n"
        "1. 論点のグルーピング（見出し＋短い説明）\n"
        "2. 各グループの代表的な意見\n"
        "3. 気づき・示唆\n"
        "4. 次に考えるとよさそうな問い\n"
    )

    user_prompt = f"ブレストの発言一覧:\n{messages_text}"

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    summary_text = generate_gemini_text(full_prompt, temperature=0.4)

    session.last_summary = summary_text

    _save_state()
    return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # 単純にこのファイルから起動したときは、リロード機能なしで
    # 直接 app オブジェクトを uvicorn に渡す。
    # （import 文字列 "app.main:app" を使うと、リロード用サブプロセスから
    #  モジュール "app" が見つからずに ModuleNotFoundError になるため）
    uvicorn.run(app, host="0.0.0.0", port=8000)


