
# Задаем процессы, которые будут запускаться из главного файла проекта
procs = {
    'transform_data': ['Подключить преобразование данных ', False], 
    'preproc_data': ['Подключить предобработку данных ', False], 
    'check_model': ['Загрузить и провести тестирование модели/-ей для talk-ai.', False],
    'server_client_test': ['Запустить сервер ', False],  
    'cli_test': ['Провести тестирование локально: ', True]
}

# Затем предобработку данных и сервер по условию выбора процесса
if __name__ == "__main__":
    try:
        for proc_name in procs:
            procs[proc_name][1] = bool(int(input(procs[proc_name][0] + '1 - да / 0 - нет: '))) 
            
        if procs['transform_data'][1] == True:
            from utils.data_schema  import data_to_json
            from data.source_data.prompts import my_qa_prompts 

            print("\nПроизводится сериализация данных для ассистента-финконсультанта.")
            path_to_srcfile = './data/source_data/prompts.py'
            data_to_json(**my_qa_prompts)  

        if procs['preproc_data'][1] == True:
            from utils import clear_data             
            
            path_to_srcfile = './data/source_data/phrases.txt'
            print("\nИдет предобработка данных для ассистента-собеседника - разметка и очистка текста, лемматизация, токенизация.")
            clear_data.data_preproc(path_to_srcfile)  
        
        if procs['check_model'][1] == True:
            from huggingface_hub import login
            from dotenv import load_dotenv
            import os

            from config.constants import MLFLOW_URL
            from config import config 
            from utils.check_model import test_model


            print('Загрузка и тестирование выбранной модели с регистрацией параметров эксперимента.\nДля просмотра результатов необходимо запустить mlflow-сервер: "http://localhost:5000"')   
            load_dotenv()
            token = os.getenv('HF_READ')
            login(token=token)

            mlflow_url = MLFLOW_URL             # в cli запустить локально  mlflow-сервер командой: mlflow server -p 5000

            models = config.models_to_train     # выбор списка моделей для загрузки и тестирования
            model_tag = config.exp_tags[0]      # выбор типа эксперимента для регистрации в mlflow: загрузка, предварительное тестирование, обучение и так далее
            exp_name = model_tag.capitalize()   # имя эксперимента

            print('exp_name: ', exp_name)

            test_model(models, exp_name, mlflow_url)     

        if procs['cli_test'][1] == True:
            from tests.cli_ui import model_answer

            print("Идет подготовка и запуск тестирования. На Ваш запрос ответят обе модели.")
            model_answer()      

        if procs['server_client_test'][1] == True:
            import uvicorn            
            from tests.server_tmbot_ui import app
            from config.constants import LOCAL_HOST, APP_SERVER_PORT
            
            print("\nПроизводится запуск сервера.")
            uvicorn.run(app, host=LOCAL_HOST, port=APP_SERVER_PORT)     # uvicorn.run(app, host="0.0.0.0", port=8000)      
        
    except Exception as e:
        print(f'Ошибка: {e}')




















