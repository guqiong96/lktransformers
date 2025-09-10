# fast_api.py
import os
import re
import sys
import argparse
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from ktransformers.server.config.config import Config
from ktransformers.server.utils.create_interface import create_interface, GlobalInterface
from fastapi.openapi.utils import get_openapi
from ktransformers.server.api import router, post_db_creation_operations
from ktransformers.server.utils.sql_utils import Base, SQLUtil
from ktransformers.server.config.log import logger

from torch.multiprocessing import Queue 
from multiprocessing.synchronize import Event


def mount_app_routes(mount_app: FastAPI, cfg=None):
    if cfg is not None:
        mount_app.state.config = cfg
        config_singleton = Config()
        if hasattr(cfg, 'model_name'):
            config_singleton.model_name = cfg.model_name
        if hasattr(cfg, 'api_key'):
            config_singleton.api_key = cfg.api_key
        if hasattr(cfg, 'temperature'):
            config_singleton.temperature = cfg.temperature
        if hasattr(cfg, 'top_p'):
            config_singleton.top_p = cfg.top_p
        if hasattr(cfg, 'server_port'):
            config_singleton.server_port = cfg.server_port
    sql_util = SQLUtil()
    logger.info("Creating SQL tables")
    Base.metadata.create_all(bind=sql_util.sqlalchemy_engine)
    post_db_creation_operations()
    mount_app.include_router(router)


def create_app(cfg=None):
    if cfg is None:
        cfg = Config()
    if(hasattr(GlobalInterface.interface, "lifespan")):
        app = FastAPI(lifespan=GlobalInterface.interface.lifespan)
    else:
        app = FastAPI()
    if cfg.web_cross_domain:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app, cfg)
    if cfg.mount_web:
        mount_index_routes(app)
    return app


def update_web_port(config_file: str):
    ip_port_pattern = (
        r"(localhost|((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)):[0-9]{1,5}"
    )
    with open(config_file, "r", encoding="utf-8") as f_cfg:
        web_config = f_cfg.read()
    ip_port = "localhost:" + str(Config().server_port)
    new_web_config = re.sub(ip_port_pattern, ip_port, web_config)
    with open(config_file, "w", encoding="utf-8") as f_cfg:
        f_cfg.write(new_web_config)


def mount_index_routes(app: FastAPI):
    # Adjust the path since this file is now in the same directory as main.py
    project_dir = os.path.dirname(__file__)
    web_dir = os.path.join(project_dir, "website/dist")
    web_config_file = os.path.join(web_dir, "config.js")
    update_web_port(web_config_file)
    if os.path.exists(web_dir):
        app.mount("/web", StaticFiles(directory=web_dir), name="static")
    else:
        err_str = f"No website resources in {web_dir}, please compile the website by npm first"
        logger.error(err_str)
        print(err_str)
        exit(1)


def custom_openapi(app):
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ktransformers server",
        version="1.0.0",
        summary="This is a server that provides a RESTful API for ktransformers.",
        description="We provided chat completion and openai assistant interfaces.",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {"url": "https://kvcache.ai/media/icon_1.png"}
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def start_fast_api(cfg, args, generated_token_queue:Queue = None, start_event: Event = None, kvcache_event: Event = None):
    
    interface = create_interface(config=cfg, default_args=cfg)
    
    setattr(interface, "token_queue", generated_token_queue)
    setattr(interface, "start_event", start_event)
     
    app = create_app(cfg)
    custom_openapi(app)
    
    def start_queue_proxy():
        interface.run_queue_proxy()
    import threading           
    proxy_thread = threading.Thread(target=start_queue_proxy, daemon=True)
    proxy_thread.start()
    
    # Run the server
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")


if __name__ == "__main__":
    pass