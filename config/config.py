#модели
models_longlist = {
    'RuGPT-3': ['ai-forever/rugpt3large_based_on_gpt2', 'talk'],
    'RuGPT-3-small': ['ai-forever/rugpt3small_based_on_gpt2', 'talk'],
    'YandexGPT': ['yandex/YandexGPT-5-Lite-8B-pretrain', 'talk'],
    'GigaChat': ['ai-forever/ruGPT-3.5-13B', 'fincon'],    
    'T-Pro-lite': ['t-tech/T-lite-it-1.0', 'fincon'],
    'T-Pro': ['t-tech/T-pro-it-1.0', 'fincon'],
}

models_to_train = {
    # 'RuGPT-3-small': ['.models/talk-ai/models--ai-forever--rugpt3small_based_on_gpt2', 'all'],  # for local_load
    'RuGPT-3-small': ['ai-forever/rugpt3small_based_on_gpt2', 'all'],  #  for load from HF
}

# Тэги экспериментов
exp_tags = ['pre-test', 'train', 'test']
