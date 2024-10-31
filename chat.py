import litserve as ls
from model import LlamaAPI

if __name__ == "__main__":
    llama_api = LlamaAPI()
    server = ls.LitServer(
        llama_api,
        spec=ls.OpenAISpec(),
        timeout=30
    )
    server.run(generate_client_file=False)
