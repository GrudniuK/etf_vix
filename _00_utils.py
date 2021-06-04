

#from datetime import datetime
import os

def prepare_directory(path_in: str) -> str:
    """
    funkcja tworzy folder o nazwie daty w formacie YYYYMMDD i zwraca sciezke do tego folderu
    
    Parameters
    ----------
    path: str
        wzgledna sciezka do folderu
    Returns
    -------
    path_out: str
        wzgledna sciezka do folderud
    """
    #current_date = datetime.date(datetime.now()).strftime('%Y%m%d')
    if not os.path.exists(path_in):
        os.makedirs(path_in)
    path_out = './'+ path_in
    return path_out
