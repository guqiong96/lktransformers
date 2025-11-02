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
 
def signal_handler(signum, frame):
    print(f"Received signal {signum}, shutting down...")
    cleanup(sched_process, fastapi_process)
    os._exit(0)

def cleanup(sched_process, fastapi_process): 
    if sched_process and sched_process.poll() is None:
        print("Terminating scheduler process...")
        try:
            sched_process.terminate() 
            sched_process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            print("Scheduler process did not terminate in time, killing it...")
            try:
                sched_process.kill()
            except Exception as e:
                print(f"Error killing scheduler process: {e}")

    if fastapi_process and fastapi_process.is_alive():
        print("Terminating FastAPI process...")
        try:
            fastapi_process.terminate() 
            fastapi_process.join(timeout=5.0)
        except Exception as e:
            print(f"Error terminating FastAPI process: {e}") 
            if fastapi_process.is_alive():
                try:
                    os.kill(fastapi_process.pid, signal.SIGKILL)
                except Exception as e2:
                    print(f"Error killing FastAPI process: {e2}")


def monitor_processes(sched_process, fastapi_process): 
    while True: 
        if sched_process and sched_process.poll() is not None:
            print(f"Scheduler process exited with code {sched_process.returncode}") 
            cleanup(sched_process, fastapi_process)
            print("Exiting main process due to scheduler failure")
            os._exit(1)
            break

        if fastapi_process and not fastapi_process.is_alive():
            print("FastAPI process is not alive") 
            cleanup(sched_process, fastapi_process)
            print("Exiting main process due to FastAPI failure")
            os._exit(1)
            break

        import time
        time.sleep(5)   

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
     
    global sched_process, fastapi_process
    sched_process = subprocess.Popen(
        ["python3", target_file, "--config", temp_file_path],
        stdout=log,
        stderr=log
    )
    print("sched_rpc started with PID:", sched_process.pid) 
     
    fastapi_process = ctx.Process(target=start_fast_api, args=(cfg, args, token_queue, start_event, kvcache_event))
    fastapi_process.daemon = True
    fastapi_process.start()
    print("fastapi started with PID:", fastapi_process.pid)
     
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
       
    import threading
    monitor_thread = threading.Thread(target=monitor_processes, args=(sched_process, fastapi_process), daemon=True)
    monitor_thread.start()
     
    try: 
        run_engine(cfg, token_queue, broadcast_endpoint, start_event, kvcache_event)
    except KeyboardInterrupt:
        print("Received interrupt signal, shutting down...")
    except Exception as e:
        import traceback
        traceback.print_exc()
        cleanup(sched_process, fastapi_process)
        sys.exit(1)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  
    main()