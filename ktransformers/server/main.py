# main.py
import os
import re
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn.logging
import uvicorn
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
from fastapi.middleware.cors import CORSMiddleware
from ktransformers.server.args import ArgumentParser
from ktransformers.server.config.config import Config
from ktransformers.server.utils.create_interface import create_interface, GlobalInterface
from fastapi.openapi.utils import get_openapi
from ktransformers.server.api import router, post_db_creation_operations
from ktransformers.server.utils.sql_utils import Base, SQLUtil
from ktransformers.server.config.log import logger
import torch.multiprocessing as mp
import tempfile
import pickle
import subprocess
from ktransformers.server.backend.interfaces.balance_serve import run_engine
import torch.multiprocessing as mp
from multiprocessing import Process, Event
from torch.multiprocessing import Queue
import signal
from ktransformers.server.fast_api import start_fast_api
 

def main():  
    cfg = Config()
    arg_parser = ArgumentParser(cfg)
    args = arg_parser.parse_args()
    for key, value in vars(args).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
     
    if hasattr(args, 'device') and args.device.startswith('cuda'):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    ctx = mp.get_context("spawn")
    token_queue = ctx.Queue(maxsize=1000)
    broadcast_endpoint = tempfile.NamedTemporaryFile(delete=False).name
    start_event = ctx.Event()
    kvcache_event = ctx.Event() 
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        pickle.dump(cfg, temp_file)
        temp_file_path = temp_file.name 
        
    current_file = __file__
    target_file = os.path.join(os.path.dirname(current_file), "balance_serve", "sched_rpc.py")
    target_file = os.path.normpath(target_file)
     
    log_path = os.path.join(args.log_dir, "rpc.log")
    log = open(log_path, "a")
     
    sched_process = subprocess.Popen(
        ["python3", target_file, "--config", temp_file_path],
        stdout=log,
        stderr=log
    )
    print("sched_rpc started with PID:", sched_process.pid) 
     
    fastapi_process = ctx.Process(target=start_fast_api, args=(cfg, args, token_queue, start_event, kvcache_event))
    fastapi_process.start()
    print("fastapi started with PID:", fastapi_process.pid)
      
    def signal_handler(signum, frame):
        print(f"Received signal {signum}, shutting down...")
        cleanup()
        os._exit(0)

    def cleanup():
        print("Cleaning up...")
        if sched_process and sched_process.poll() is None:
            print(f"Terminating sched_process {sched_process.pid}")
            sched_process.terminate()
            sched_process.wait()
        if fastapi_process:
            print(f"Terminating api_process {fastapi_process.pid}")
            fastapi_process.terminate() 
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
     
    try: 
        run_engine(cfg, token_queue, broadcast_endpoint, start_event, kvcache_event)
    except KeyboardInterrupt:
        print("Received interrupt signal, shutting down...")
    except Exception as e:
        print(f"Engine error: {e}")
    finally: 
        cleanup()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  
    main()