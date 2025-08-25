import sys

from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletion
from rich.console import Console
from rich.markdown import Markdown
from ttg.colors import FONT_RESET, BOLD_GOLD


# Full list of models is available at https://console.groq.com/docs/models
MODELS = {
    "compound": "compound-beta",
    "cm": "compound-beta-mini",
    'scout': 'meta-llama/llama-4-scout-17b-16e-instruct',
    '3-70b': 'llama-3.3-70b-versatile',
    # 'gemma': 'gemma2-9b-it',  # deprecated on 08/10/2025
    # 'mistral': 'mistral-saba-24b',  # deprecated on 30/07/2025
    'qwen': 'qwen/qwen3-32b',
    'instant': 'llama-3.1-8b-instant',
}


class ChatGroq:

    def __init__(self):
        self.model=MODELS['scout']
        self.user_font = BOLD_GOLD
        self.model_font = BOLD_GOLD
        self.client = Groq()
        self.user = "Paweł"
        self.user_pretty_prompt = f'{self.user_font}{self.user}{FONT_RESET}: '
        self.model_pretty_prompt = f'{self.model_font}{self.model}{FONT_RESET}: '
        self.messages = []
        self.console = Console()


    def get_user_prompt(self) -> bool:
        try:
            message = input(self.user_pretty_prompt)
        except (KeyboardInterrupt, EOFError):
            print('')
            return False

        if not message:
            return False

        self.messages.append({
            'role': 'user',
            'content': message,
        })
        return True


    def call_the_model(self):
        create = self.client.chat.completions.create
        chat_completion: ChatCompletion = create(
            messages=self.messages,
            model=self.model,
            max_completion_tokens=8192,  # default is 1024
        )
        message = chat_completion.choices[0].message
        self.messages.append(message)


    def print_message(self):
        content = self.messages[-1].content
        print(self.model_pretty_prompt)
        if content is not None:
            message_md = Markdown(content)
            self.console.print(message_md)
        else:
            print("Error, groq returned None.", file=sys.stderr)


def main():
    load_dotenv()
    chat = ChatGroq()
    while True:
        oll_korrect = chat.get_user_prompt()
        if not oll_korrect:
            break

        chat.call_the_model()
        chat.print_message()


if __name__ == "__main__":
    main()
