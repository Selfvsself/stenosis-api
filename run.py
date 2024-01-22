import uvicorn
import torch

from multiprocessing import Process


def run_localhost():
    uvicorn.run('main:app', port=8085, host="0.0.0.0")


if __name__ == '__main__':
    run_localhost_proc = Process(target=run_localhost)

    run_localhost_proc.start()
