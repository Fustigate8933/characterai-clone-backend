import litserve as ls
from model import LlamaAPI

if __name__ == "__main__":
    api = LlamaAPI()
    server = ls.LitServer(
        api,
        spec=ls.OpenAISpec(),
        timeout=30
    )
    server.run(generate_client_file=False)
