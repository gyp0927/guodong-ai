import tkinter as tk
from tkinter import ttk, messagebox
import os
import subprocess
import sys
import dotenv

# 支持的提供商
PROVIDERS = {
    # 本地
    "ollama": {
        "name": "本地 Ollama（免费）",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.2",
        "need_api_key": False,
        "models": ["llama3.2", "llama3.1:8b", "llama3.1:70b", "mistral", "codellama", "qwen2.5", "deepseek-r1:7b"]
    },
    # 国内
    "deepseek": {
        "name": "DeepSeek（国内）",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "need_api_key": True,
        "models": ["deepseek-chat", "deepseek-reasoner", "deepseek-coder"]
    },
    "qwen": {
        "name": "阿里 Qwen（国内）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "need_api_key": True,
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-coder-plus"]
    },
    "minimax": {
        "name": "MiniMax（国内）",
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "MiniMax-Text-01",
        "need_api_key": True,
        "models": ["MiniMax-Text-01", "abab7-chat-preview"]
    },
    "doubao": {
        "name": "字节豆包（国内）",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "doubao-pro-32k",
        "need_api_key": True,
        "models": ["doubao-pro-32k", "doubao-pro-128k", "doubao-lite-32k"]
    },
    "glm": {
        "name": "智谱 GLM（国内）",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
        "need_api_key": True,
        "models": ["glm-4-flash", "glm-4-plus", "glm-4-air", "glm-4v"]
    },
    "ernie": {
        "name": "百度文心（国内）",
        "base_url": "https://qianfan.baidubce.com/v1",
        "default_model": "ernie-4.0-8k-latest",
        "need_api_key": True,
        "models": ["ernie-4.0-8k-latest", "ernie-4.0-turbo-8k", "ernie-3.5-8k", "ernie-speed-8k"]
    },
    "hunyuan": {
        "name": "腾讯混元（国内）",
        "base_url": "https://hunyuan.tencentcloudapi.com/v1",
        "default_model": "hunyuan-pro",
        "need_api_key": True,
        "models": ["hunyuan-pro", "hunyuan-standard", "hunyuan-lite"]
    },
    "spark": {
        "name": "讯飞星火（国内）",
        "base_url": "https://spark-api.xf-yun.com/v3.1",
        "default_model": "spark-4.0",
        "need_api_key": True,
        "models": ["spark-4.0", "spark-3.5-max", "spark-lite"]
    },
    "kimi": {
        "name": "月之暗面 Kimi（国内）",
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "kimi-k2.6",
        "need_api_key": True,
        "models": ["kimi-k2.6", "kimi-k2", "kimi-k1.5", "kimi-latest", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]
    },
    "kimi-code": {
        "name": "Kimi Code（国内 · 免费）",
        "base_url": "https://api.kimi.com/coding/v1",
        "default_model": "kimi-for-coding",
        "need_api_key": True,
        "models": ["kimi-for-coding"]
    },
    "yi": {
        "name": "零一万物 Yi（国内）",
        "base_url": "https://api.lingyiwanwu.com/v1",
        "default_model": "yi-large",
        "need_api_key": True,
        "models": ["yi-large", "yi-medium", "yi-spark"]
    },
    "baichuan": {
        "name": "百川 Baichuan（国内）",
        "base_url": "https://api.baichuan-ai.com/v1",
        "default_model": "baichuan4",
        "need_api_key": True,
        "models": ["baichuan4", "baichuan4-turbo", "baichuan3-turbo"]
    },
    # 国外
    "openai": {
        "name": "OpenAI（国外）",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "need_api_key": True,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini", "o3-mini"]
    },
    "anthropic": {
        "name": "Anthropic（国外）",
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-sonnet-4-20250514",
        "need_api_key": True,
        "models": ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"]
    },
    "gemini": {
        "name": "Google Gemini（国外）",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash",
        "need_api_key": True,
        "models": ["gemini-2.0-flash", "gemini-2.5-pro-preview-03-25", "gemini-1.5-pro", "gemini-1.5-flash"]
    },
    "grok": {
        "name": "xAI Grok（国外）",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-3",
        "need_api_key": True,
        "models": ["grok-3", "grok-3-mini", "grok-2"]
    },
    "mistral": {
        "name": "Mistral AI（国外）",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
        "need_api_key": True,
        "models": ["mistral-large-latest", "mistral-small-latest", "codestral-latest", "pixtral-large-latest"]
    },
    "cohere": {
        "name": "Cohere（国外）",
        "base_url": "https://api.cohere.com/compatibility/v1",
        "default_model": "command-r-plus",
        "need_api_key": True,
        "models": ["command-r-plus", "command-r", "command-r7b-12-2024"]
    },
    "perplexity": {
        "name": "Perplexity（国外）",
        "base_url": "https://api.perplexity.ai",
        "default_model": "sonar-pro",
        "need_api_key": True,
        "models": ["sonar-pro", "sonar-reasoning", "sonar-deep-research", "sonar"]
    },
    "groq": {
        "name": "Groq（国外）",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "need_api_key": True,
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
    },
    "together": {
        "name": "Together AI（国外）",
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "need_api_key": True,
        "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.1-405B-Instruct-Turbo", "deepseek-ai/DeepSeek-R1"]
    },
}


def load_existing_config():
    """加载现有配置"""
    dotenv.load_dotenv()
    return {
        "provider": os.getenv("LLM_PROVIDER", "ollama"),
        "api_key": os.getenv("LLM_API_KEY", ""),
        "model": os.getenv("LLM_MODEL_NAME", ""),
    }


def save_config(provider, api_key, model):
    """保存配置到 .env"""
    config = f'''# LLM 配置
LLM_PROVIDER={provider}
LLM_API_KEY={api_key}
LLM_BASE_URL={PROVIDERS[provider]["base_url"]}
LLM_MODEL_NAME={model}
'''
    with open(".env", "w", encoding="utf-8") as f:
        f.write(config)


def launch_server():
    """启动 Flask 服务器"""
    python = sys.executable
    subprocess.Popen([python, "web/app.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)


def wait_for_server(url="http://127.0.0.1:5000", timeout=30):
    """等待服务器启动"""
    import time
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


class ConfigGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("果冻ai - 配置")
        self.window.geometry("600x500")
        self.window.resizable(False, False)

        # 居中显示
        self.window.update_idletasks()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - 600) // 2
        y = (screen_height - 500) // 2
        self.window.geometry(f"600x500+{x}+{y}")

        self.existing_config = load_existing_config()

        self.create_widgets()
        self.load_config()

    def create_widgets(self):
        # 标题
        title = tk.Label(self.window, text="🤖 果冻ai 配置", font=("Arial", 18, "bold"))
        title.pack(pady=20)

        # 选择提供商
        frame_provider = tk.Frame(self.window)
        frame_provider.pack(fill="x", padx=40, pady=10)
        tk.Label(frame_provider, text="选择模型提供商：", font=("Arial", 11)).pack(anchor="w")

        self.provider_var = tk.StringVar()
        self.provider_combo = ttk.Combobox(frame_provider, textvariable=self.provider_var, state="readonly", font=("Arial", 11))
        self.provider_combo["values"] = [f"{k} - {v['name']}" for k, v in PROVIDERS.items()]
        self.provider_combo.pack(fill="x", pady=5)
        self.provider_combo.bind("<<ComboboxSelected>>", self.on_provider_changed)

        # 输入 API Key
        frame_key = tk.Frame(self.window)
        frame_key.pack(fill="x", padx=40, pady=10)
        tk.Label(frame_key, text="API Key：", font=("Arial", 11)).pack(anchor="w")

        self.api_key_var = tk.StringVar()
        self.api_key_entry = tk.Entry(frame_key, textvariable=self.api_key_var, font=("Arial", 11), show="*")
        self.api_key_entry.pack(fill="x", pady=5)

        self.api_key_hint = tk.Label(frame_key, text="", font=("Arial", 9), fg="gray")
        self.api_key_hint.pack(anchor="w")

        # 选择模型
        frame_model = tk.Frame(self.window)
        frame_model.pack(fill="x", padx=40, pady=10)
        tk.Label(frame_model, text="选择模型：", font=("Arial", 11)).pack(anchor="w")

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(frame_model, textvariable=self.model_var, state="readonly", font=("Arial", 11))
        self.model_combo.pack(fill="x", pady=5)

        # 自定义模型输入
        frame_custom = tk.Frame(self.window)
        frame_custom.pack(fill="x", padx=40, pady=5)
        tk.Label(frame_custom, text="或输入自定义模型名称：", font=("Arial", 9), fg="gray").pack(anchor="w")

        self.custom_model_var = tk.StringVar()
        self.custom_model_entry = tk.Entry(frame_custom, textvariable=self.custom_model_var, font=("Arial", 11))
        self.custom_model_entry.pack(fill="x", pady=5)

        # 按钮
        frame_buttons = tk.Frame(self.window)
        frame_buttons.pack(pady=30)

        self.launch_btn = tk.Button(frame_buttons, text="🚀 启动聊天", font=("Arial", 14, "bold"),
                                     bg="#4CAF50", fg="white", padx=30, pady=10, cursor="hand2",
                                     command=self.launch)
        self.launch_btn.pack(side="left", padx=10)

        self.save_btn = tk.Button(frame_buttons, text="💾 保存配置", font=("Arial", 12),
                                   bg="#2196F3", fg="white", padx=20, pady=10, cursor="hand2",
                                   command=self.save_only)
        self.save_btn.pack(side="left", padx=10)

        # 底部提示
        self.hint_label = tk.Label(self.window, text="", font=("Arial", 9), fg="#888")
        self.hint_label.pack(side="bottom", pady=10)

    def load_config(self):
        provider = self.existing_config["provider"]
        for p in PROVIDERS:
            if p == provider:
                self.provider_var.set(f"{p} - {PROVIDERS[p]['name']}")
                break
        self.on_provider_changed()
        self.api_key_var.set(self.existing_config["api_key"])
        if self.existing_config["model"]:
            self.model_var.set(self.existing_config["model"])

    def on_provider_changed(self, event=None):
        selection = self.provider_var.get()
        if not selection:
            return
        provider_key = selection.split(" - ")[0]
        provider = PROVIDERS.get(provider_key)

        if provider:
            # 更新模型列表
            self.model_combo["values"] = provider["models"]
            self.model_var.set(provider["default_model"])

            # 更新 API Key 提示
            if provider["need_api_key"]:
                self.api_key_hint.config(text=f"需要 API Key，请从 {provider['name']} 官网获取")
                self.api_key_entry.config(state="normal")
            else:
                self.api_key_hint.config(text="Ollama 不需要 API Key，可留空或填写 'ollama'")
                self.api_key_var.set("ollama")
                self.api_key_entry.config(state="normal")

    def get_selected_model(self):
        custom = self.custom_model_var.get().strip()
        if custom:
            return custom
        return self.model_var.get()

    def validate(self):
        selection = self.provider_var.get()
        if not selection:
            messagebox.showwarning("提示", "请选择模型提供商")
            return False

        provider_key = selection.split(" - ")[0]
        provider = PROVIDERS.get(provider_key)

        if provider and provider["need_api_key"]:
            api_key = self.api_key_var.get().strip()
            if not api_key:
                messagebox.showwarning("提示", "请输入 API Key")
                return False

        if not self.get_selected_model():
            messagebox.showwarning("提示", "请选择或输入模型名称")
            return False

        return True

    def save_only(self):
        if not self.validate():
            return
        selection = self.provider_var.get()
        provider_key = selection.split(" - ")[0]
        save_config(provider_key, self.api_key_var.get().strip(), self.get_selected_model())
        messagebox.showinfo("成功", "配置已保存！")

    def launch(self):
        if not self.validate():
            return

        selection = self.provider_var.get()
        provider_key = selection.split(" - ")[0]

        # 保存配置
        save_config(provider_key, self.api_key_var.get().strip(), self.get_selected_model())

        # 关闭窗口
        self.window.destroy()

        # 显示启动提示
        print("正在启动服务器，请稍候...")

        # 启动服务器
        launch_server()

        # 等待服务器就绪
        if wait_for_server():
            print("服务器已就绪，正在打开浏览器...")
            import webbrowser
            webbrowser.open("http://127.0.0.1:5000")
        else:
            print("服务器启动可能需要更长时间，请手动访问 http://127.0.0.1:5000")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    # 确保在正确目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = ConfigGUI()
    app.run()
