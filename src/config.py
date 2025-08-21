import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()

def get_key(key_name):
    return os.getenv(key_name)

if __name__ == "__main__":
    load_env()
    print("API_KEY present:", get_key("API_KEY") is not None)


