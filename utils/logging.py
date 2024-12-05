import os
import datetime
import logging


def getLogger(*args, **kwargs):
    return logging.getLogger(*args, **kwargs)

def setup_logging(*, outpath:str=None, jobname:str='test', verbose:bool=False) -> None:
    logging_level = logging.DEBUG if verbose else logging.INFO
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(outpath, f"{time}_{jobname}.log") if outpath is not None else None
    logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s\n%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging_level,
                        handlers=[
                            logging.FileHandler(filepath),
                            logging.StreamHandler()]
                        )