from waitress import serve
from voice import application as voicer
from app import application as server

if __name__ == '__main__':
    serve(server, listen='127.0.0.1:8000')
    