import litserve as ls
from model import TTSAPI

if __name__ == "__main__":
    tts_api = TTSAPI()
    server = ls.LitServer(
        tts_api,
        # timeout=30
    )
    server.run(generate_client_file=False, port=8001)
