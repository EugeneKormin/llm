import configparser
config = configparser.ConfigParser()


config.read('D:\my_apps\python_app\llm\config.ini')

OPENAI_API_TOKEN = config['OPENAI']['API_TOKEN']
