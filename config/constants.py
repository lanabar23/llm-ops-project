# Actual Remote host IP 
REMOTE_HOST_KLAVDIA = '176.108.250.95' # это PUB_IP VM в СберКлаудии. Если сервер развернут на локальной машине, то меняем на 'localhost'.

# Задание хостов для тестирования соединения
LOCAL_HOST =  'localhost'
REMOTE_HOST = REMOTE_HOST_KLAVDIA
PING_HOST = REMOTE_HOST

# Порт и хост Mlflow ервера
MLFLOW_SERVER_PORT = 5000
MLFLOW_SERVER_HOST = LOCAL_HOST

#Порт приложения
APP_SERVER_PORT = 8080

# Полный URL для local и remote endpoint app
MLFLOW_URL = f'http://{LOCAL_HOST}:{MLFLOW_SERVER_PORT}'
LOCAL_APP_URL = f"http://{LOCAL_HOST}:{APP_SERVER_PORT}"
REMOTE_APP_URL = f"http://{REMOTE_HOST}:{APP_SERVER_PORT}"

# Пути к исходным данным и тестам
PATH_TO_TALKDATA = './data/processed_data/phrases.csv'

PATH_TO_FINDATA = './data/processed_data/finai_data.json'

