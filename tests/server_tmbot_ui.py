from sentence_transformers import SentenceTransformer
# from langfuse import Langfuse, observe, get_client
import time, json, os, logging, faiss
from fastapi import FastAPI, Request
from transformers import pipeline
from dotenv import load_dotenv


from config.constants import PATH_TO_FINDATA

load_dotenv()
# lf_pubkey = os.getenv('LF_PUBKEY')
# lf_seckey = os.getenv('LF_SECKEY')
max_distance =  1
best_distance = 0.29 # 0.5
top_k = 1

logger = logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Инициализация FastAPI-приложения
app = FastAPI()

# Инициализация langfuse - соединения
# try:
#     client = Langfuse(
#     public_key=lf_pubkey,
#     secret_key=lf_seckey,
#     host="https://cloud.langfuse.com"           
#     )
    # client.run_experiment()
    # langfuse = get_client()
# except Exception as e:
#     print(f'Langfuse не подключен. Ошибка {e}')

path_to_file = PATH_TO_FINDATA 
with open(path_to_file, 'r', encoding='utf-8') as f:
    data = json.load(f)['prompts']

# Загрузка генератора текста на основе RuGPT3small
talk_model_name = "sberbank-ai/rugpt3small_based_on_gpt2" # models_to_train['RuGPT-3-small'][0]
generator = pipeline('text-generation', model=talk_model_name)

# Загрузка модели для финконсультанта на оснвое BERT
consult_model_name = 'sberbank-ai/sbert_large_nlu_ru' # 'DeepPavlov/rubert-base-cased' 
bert_model = SentenceTransformer(consult_model_name)

questions = list(data.values())
model = bert_model  # Используем вашу модель
question_embeddings = model.encode([item['question'] for item in questions])

dim = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(question_embeddings.astype('float32'))
ids = list(data.keys())  # сохраняем ключи заранее


@app.post("/generate")
async def generate_text(request: Request):
    req_data = await request.json()  # Получаем тело запроса
    print("generate/req_data: ", req_data)
    input_text = req_data["input"]

    # Генерируем текст с помощью модели talk_model_name
    output = generator(input_text, max_length=5, max_new_tokens=20, temperature=0.7)[0]["generated_text"]

    return {"output": output}

# @observe()  
@app.post("/consult")
async def choice_answer(request: Request):
    req_data = await request.json()  # Получаем тело запроса    
    input_text = req_data["input"]

    start_time = time.time()

    # Производим поиск ближайшего вектора с помощью bert_model
    query_embedding = bert_model.encode(input_text)
    D, I = index.search(query_embedding.reshape(1, dim), k=top_k)
    if D <= best_distance: 
        best_match_idx = I[0][0]
        best_match_id = ids[best_match_idx]
        result = data.get(best_match_id)['finai_answer']
    else:
        result = 'Спасибо, что обратились ко мне за советом, но я не дам Вам ответа.'

    end_time = time.time()
    execution_time = end_time - start_time
    
    # Вывод в терминал
    print("Клиентский запрос: ", req_data['input'])
    print('Ответ модели:\n', result)
    print(f'Время ответа модели: {execution_time:.2f}')

    # langfuse.flush()

    return {"output": result}  
