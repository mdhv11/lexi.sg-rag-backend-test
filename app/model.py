from llama_cpp import Llama


class LocalLLM:
    def __init__(self, model_path="llama-2-7b-chat.ggmlv3.q4_0.bin"):
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=6)

    def generate(self, prompt):
        output = self.llm(prompt, max_tokens=300, stop=["</s>"])
        return output["choices"][0]["text"].strip()
