from pydantic import BaseModel
from typing import Dict
import json

from config.constants import PATH_TO_FINDATA

# Модель структуры данных для финконсультанта
class QAItem(BaseModel):
    question: str
    finai_answer: str
    url: str

class MyQAPrompts(BaseModel):
    prompts: Dict[str, QAItem]


def data_to_json(**kwargs):
    
    # print(kwargs)
     # Создание экземпляра модели
    model_instance = MyQAPrompts(prompts={k: QAItem(**v) for k, v in kwargs.items()})

    # Экспорт в JSON-файл
    with open(PATH_TO_FINDATA, 'w', encoding='utf-8') as f:
        f.write(model_instance.model_dump_json(indent=1))


