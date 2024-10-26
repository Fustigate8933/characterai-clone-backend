from litserve.specs.openai import ChatCompletionRequest


def parse_messages(request: ChatCompletionRequest):
    messages = []
    tools = request.tools
    n = len(request.messages)
    for i, message in enumerate(request.messages):
        
