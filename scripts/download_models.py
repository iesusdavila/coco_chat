from llama_cpp import Llama

llm = Llama.from_pretrained(
    # repo_id: It is the repository id of the model you want to download
	repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    # filename: It is the name of the file in which the model will be downloaded
	filename="tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
)