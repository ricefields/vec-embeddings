
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large" 
#EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl"
#EMBEDDING_MODEL_NAME =  "intfloat/e5-large-v2"
#EMBEDDING_MODEL_NAME =  "intfloat/e5-base-v2"

LLM_MODEL_LOCAL = 1
LLM_MODEL_OPENAI = 0     # OpenAI for inferencing
LLM_MODEL_BEDROCK = 0    # AWS Bedrock for inferencing

LLM_EMBEDDINGS_LOCAL = 1 # Use HuggingsFaceEmbeddings
LLM_EMBEDDINGS_OPENAI = 0
LLM_EMBEDDINGS_BEDROCK = 0

LOCAL_MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
LOCAL_MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"

STREAMLIT_DEBUG_MODE = 1