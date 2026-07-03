"""
WeChat ClawBot for Claude — Python implementation.
Based on @tencent-weixin/openclaw-weixin iLink protocol.
Direct QR code login, no OpenClaw needed. Forwards messages to Claude via Anthropic API.
"""
import json, os, base64, struct, time, logging, threading, sys
import requests
from http.client import HTTPSConnection
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wechat-claude")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "configs", "wechat_claude_config.json")
SESSION_FILE = os.path.join(os.path.dirname(__file__), "..", "configs", "wechat_session.json")

DEFAULT_CONFIG = {
    "anthropic_api_key": "",
    "anthropic_model": "claude-sonnet-4-6",
    "system_prompt": "你是一个友好的AI助手。回复要简洁、自然，用中文。",
    "max_tokens": 1024,
    "temperature": 0.7,
}

# ── WeChat iLink Protocol Implementation ────────────────────

IINK_BASE = "ilinkai.weixin.qq.com"
def b64_uin():
    return base64.b64encode(struct.pack("<I", int(time.time() * 1000) & 0xFFFFFFFF)).decode()

class WeChatSession:
    def __init__(self, token, uin):
        self.token = token
        self.uin = uin
        self.updates_buf = ""

    def headers(self):
        return {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "Authorization": f"Bearer {self.token}",
            "X-WECHAT-UIN": b64_uin(),
        }

    def get_updates(self, timeout=35):
        """Long-poll for new messages."""
        conn = HTTPSConnection(IINK_BASE, timeout=timeout + 10)
        try:
            body = json.dumps({"get_updates_buf": self.updates_buf})
            conn.request("POST", "/getupdates", body, self.headers())
            resp = conn.getresponse()
            data = json.loads(resp.read())
            if data.get("ret") == 0:
                self.updates_buf = data.get("get_updates_buf", self.updates_buf)
                return data.get("msgs", [])
            return []
        except Exception as e:
            log.warning(f"get_updates error: {e}")
            return []
        finally:
            conn.close()

    def send_text(self, to_user_id, context_token, text):
        """Send text message."""
        conn = HTTPSConnection(IINK_BASE, timeout=15)
        try:
            body = json.dumps({
                "msg": {
                    "to_user_id": to_user_id,
                    "context_token": context_token,
                    "item_list": [{"type": 1, "text_item": {"text": text}}],
                }
            })
            conn.request("POST", "/sendmessage", body, self.headers())
            resp = conn.getresponse()
            return json.loads(resp.read())
        finally:
            conn.close()

    def send_typing(self, to_user_id, context_token, typing=True):
        """Show/hide typing indicator."""
        conn = HTTPSConnection(IINK_BASE, timeout=10)
        try:
            body = json.dumps({
                "msg": {
                    "to_user_id": to_user_id,
                    "context_token": context_token,
                    "item_list": [{"type": 1, "text_item": {"text": ""}}],
                    "action_type": 1 if typing else 2,
                }
            })
            conn.request("POST", "/sendtyping", body, self.headers())
        finally:
            conn.close()


# ── QR Code Login ───────────────────────────────────────────

def qr_login():
    """Open browser to get QR code, then poll for login."""
    log.info("正在启动微信扫码登录...")
    # Use openclaw-weixin-cli for login if available, else manual QR
    import subprocess, tempfile
    # Generate login URL
    conn = HTTPSConnection(IINK_BASE, timeout=15)
    conn.request("POST", "/qrcode", json.dumps({"appid": "wx_bot"}), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()

    if data.get("ret") != 0:
        log.error(f"获取二维码失败: {data}")
        return None, None

    qr_url = data.get("qrcode_url", "")
    session_id = data.get("session_id", "")

    # Show QR code URL
    log.info(f"请用微信扫码登录:")
    log.info(f"如果终端不支持显示二维码，请复制链接到浏览器打开:")
    log.info(f"{qr_url}")

    # Try to render QR in terminal
    try:
        import qrcode
        from PIL import Image
        qr = qrcode.QRCode(border=1)
        qr.add_data(qr_url)
        qr.print_ascii()
    except ImportError:
        log.info("(安装 qrcode[pil] 可在终端直接显示二维码)")

    # Poll for login result
    timeout = 120
    start = time.time()
    while time.time() - start < timeout:
        conn = HTTPSConnection(IINK_BASE, timeout=15)
        conn.request("POST", "/qrcode/confirm", json.dumps({"session_id": session_id}), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()

        if data.get("ret") == 0 and data.get("token"):
            log.info("微信登录成功！")
            return data["token"], data.get("uin", b64_uin())
        time.sleep(2)

    log.error("登录超时")
    return None, None


# ── Claude API Call ─────────────────────────────────────────

def claude_reply(api_key, model, system, messages):
    """Call Claude API and return reply text."""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "system": system,
                "max_tokens": 1024,
                "temperature": 0.7,
                "messages": messages,
            },
            timeout=60,
        )
        data = resp.json()
        if "content" in data:
            return "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
        log.error(f"Claude API error: {data}")
        return f"（AI 暂时无法回复，请稍后再试）"
    except Exception as e:
        log.error(f"Claude API exception: {e}")
        return "（回复出错）"


# ── Main Loop ────────────────────────────────────────────────

def main():
    # Load config
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        config.update(json.load(open(CONFIG_FILE, encoding="utf-8")))
    else:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        input_key = input("请输入 Anthropic API Key: ").strip()
        if not input_key:
            print("需要 API Key 才能继续")
            return
        config["anthropic_api_key"] = input_key
        config["system_prompt"] = input(f"系统提示词 [{config['system_prompt']}]: ").strip() or config["system_prompt"]
        config["anthropic_model"] = input(f"Claude 模型 [{config['anthropic_model']}]: ").strip() or config["anthropic_model"]
        json.dump(config, open(CONFIG_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if not config["anthropic_api_key"]:
        log.error("未配置 Anthropic API Key")
        return

    # Login or restore session
    token, uin = None, None
    if os.path.exists(SESSION_FILE):
        session = json.load(open(SESSION_FILE, encoding="utf-8"))
        token, uin = session.get("token"), session.get("uin")
        log.info("恢复上次登录会话")

    if not token:
        token, uin = qr_login()
        if not token:
            return
        json.dump({"token": token, "uin": uin}, open(SESSION_FILE, "w", encoding="utf-8"), indent=2)

    ws = WeChatSession(token, uin)
    log.info("Bot 已就绪，等待微信消息...")

    # Conversation memory
    conversations = {}  # user_id -> list of messages

    while True:
        msgs = ws.get_updates()
        for msg in msgs:
            user_id = msg.get("from_user_id", "")
            content = ""
            for item in msg.get("item_list", []):
                if item.get("type") == 1:
                    content = item.get("text_item", {}).get("text", "")

            if not user_id or not content:
                continue

            context_token = msg.get("context_token", "")
            log.info(f"收到消息 [{user_id}]: {content[:50]}")

            # Show typing indicator
            ws.send_typing(user_id, context_token, True)

            # Build conversation
            if user_id not in conversations:
                conversations[user_id] = [{"role": "user", "content": content}]
            else:
                conversations[user_id].append({"role": "user", "content": content})
                if len(conversations[user_id]) > 20:
                    conversations[user_id] = conversations[user_id][-20:]

            # Get Claude reply
            reply = claude_reply(
                config["anthropic_api_key"],
                config["anthropic_model"],
                config["system_prompt"],
                conversations[user_id],
            )

            # Save reply to history
            conversations[user_id].append({"role": "assistant", "content": reply})

            # Send reply
            ws.send_text(user_id, context_token, reply)
            ws.send_typing(user_id, context_token, False)
            log.info(f"回复 [{user_id}]: {reply[:50]}")

        time.sleep(0.5)

if __name__ == "__main__":
    main()
