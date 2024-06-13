import configparser
config = configparser.ConfigParser()


config.read('./config.ini')

OPENAI_API_TOKEN = config['OPEN_AI']['API_TOKEN']
