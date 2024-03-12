import logging
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parent.parent

def get_logger(logger_name:str="ucav", log_file_dir: Path=PROJECT_ROOT_DIR / "my_logs" / "my_sys_logs.log"):
    handler_file = logging.FileHandler(log_file_dir)  # stdout to file
    handler_control = logging.StreamHandler()    # stdout to console
    handler_file.setLevel('DEBUG')               # 设置级别: DEBUG 、INFO 、WARNING、ERROR、CRITICAL
    handler_control.setLevel('DEBUG')             

    selfdef_fmt = '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(selfdef_fmt)
    handler_file.setFormatter(formatter)
    handler_control.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel('DEBUG')           #设置了这个才会把debug以上的输出到控制台

    logger.addHandler(handler_file)    #添加handler
    logger.addHandler(handler_control)

    return logger

# my_logger = get_logger()

# def get_default_logger() -> logging.Logger:
#     return my_logger


if __name__ == "__main__":
    my_logger = get_logger()
    my_logger.info('info,一般的信息输出')
    my_logger.warning('waring，用来用来打印警告信息')
    my_logger.error('error，一般用来打印一些错误信息')
    my_logger.critical('critical，用来打印一些致命的错误信息，等级最高')