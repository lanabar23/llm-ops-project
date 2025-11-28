from transformers import AutoModelForCausalLM, AutoTokenizer
import  torch, mlflow, time, logging           
# from langfuse import observe
import pandas as pd

from data.source_data import prompts

mlflow_metrics = {
    'status': False,
    'params': 0,
    't_load': 0,
    'memory':  0,
    'w_memory':  0,
    't_memory':  0
}

prompts = prompts.test_prompts # ['all']

# Класс для загрузки моделей
# @observe
class ModelLoader():
    def __init__(self, model_name_or_path):
        self.time_start = time.time()
        print('model_name_or_path: ', type(model_name_or_path), model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer = self.load_model()
        self.metrics = self.model_opt()

    @mlflow.trace
    def load_model(self):
        try:
            # Загрузим модель и токенайзер
            project_cache_dir = './models/talk-ai/'
            model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, cache_dir=project_cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=project_cache_dir)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            return model, tokenizer  
        except Exception as e:
            print(f"Ошибка при загрузке модели {self.model_name_or_path}: {e}")       
        
    
    # Фиксируем данные в MLflow  с помощью трейс-декоратора 
    @mlflow.trace
    def generate_response(self, prompt):
        """
        Генерация ответа на заданный промпт
        """
        start_time = time.time()
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=50)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        latency = round(time.time() - start_time, 2)

        return {'response': generated_text, 'latency': latency}

    def model_opt(self):
        metrics = {}
        metrics['status'] = True
        metrics['params'] = round((sum(p.numel() for p in self.model.parameters())) / (10 ** 9), 2)
        metrics['t_load'] = time.time() - self.time_start  
        metrics['memory'] = metrics['params'] * 4                       # Размер параметра FP32 ~ 4 байта
        metrics['w_memory'] = metrics['memory'] * 2
        metrics['t_memory'] = metrics['memory'] * 4
        return metrics

def set_mlflow_experiment(mlflow_url, exp_name):
    try:
        mlflow.set_tracking_uri(mlflow_url)  # Установка трекинга
        try:   
            exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
            print('Эксперимент найден. Продолжаем трекинг.')
        except:        
            print(f'Эксперимент с наименованием {exp_name} не найден.\nСоздаем новый эксперимент.')
            mlflow.set_experiment(exp_name)   
            exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id   
        return exp_id
    except Exception as e:
        print('Связь с сервером не установлена. Ошибка: ', e)    


def test_model(models, exp_name, mlflow_url, mlflow_metrics=mlflow_metrics):  
    print('models: ',models) 
    print('mlflow_url: ',mlflow_url)  
    experiment_id = set_mlflow_experiment(mlflow_url, exp_name)
    print(f'experiment_id: {experiment_id}') 
    results = {}
    # Основной цикл тестирования
    with mlflow.start_run(experiment_id=experiment_id, run_name='model_params'):
        model_metrics = {}
        for model_name, details in models.items():
            with mlflow.start_run(experiment_id=experiment_id, run_name=model_name, nested=True):
                print('model_name: ', model_name)
                try:
                    loader = ModelLoader(details[0])
                    model_metrics = loader.metrics       
                    category = details[1]       
                    print('category: ', category)     
                    responces = [] 
                    total_latency = 0
                    try:
                        md_experiment_id = mlflow.get_experiment_by_name(model_name).experiment_id
                    except:
                        md_experiment_id = mlflow.set_experiment(model_name)
                    run_idx = category
                    run_name = f'{model_name}-{run_idx}'

                    print('md_experiment_id: ', md_experiment_id)
                    print('prompts: ', type(prompts), prompts)
                    with mlflow.start_run(experiment_id=md_experiment_id, run_name=run_name, nested=True):
                        print(experiment_id)                        
                        for prompt in prompts.get(category):  # , []):                                            
                            response_data = loader.generate_response(prompts[category][prompt])
                            print(f'prompt: {prompts[category][prompt]}   response_data: {response_data["response"]}')
                            responces.append({prompts[category][prompt] : response_data['response']})    
                            total_latency += response_data['latency']
                            # with mlflow.start_run(nested=True):             
                            results[model_name] = responces
                        
                        avg_latency = total_latency / len(results[model_name]) if results[model_name] else float('nan')
                        mlflow.log_metric("Average latency", avg_latency)
                    try:
                        # Регистрация модели
                        mlflow.pyfunc.log_model(name=f"{model_name}")  #, python_model=PATH_TO_MODEL)
                        print('Регистрация модели прошла успешно.')
                    except Exception as e:
                        print("Модель не зарегистрирована. Ошибка: ", e)
                except Exception as e:
                    model_metrics['status'] = False
                    print(f'Модель не загрузилась. Ошибка:\n{e}')

                metrics = model_metrics if model_metrics['status'] else mlflow_metrics
                print(f'Параметры модели: {metrics}')
                for key in metrics:                    
                    mlflow.log_metric(key, metrics[key])            
                print('логирование закончено')

                
                    