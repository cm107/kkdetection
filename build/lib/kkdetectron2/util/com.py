import sys, os, shutil, re
from typing import List, Union

from numpy.lib.arraysetops import isin


__all__ = [
    "MyDict",
    "get_args",
    "setattr_deep",
    "check_type",
    "check_type_list",
    "makedirs",
    "correct_dirpath",
    "strfind",
    "conv_str_auto",
]


class MyDict(dict):
    def get(self, key: object, _type=None, init=None, autofix: bool=False):
        val = super().get(key)
        if val is None:
            return init
        elif _type is not None:
            if isinstance(val, list):
                return [_type(x) for x in val]
            else:
                return _type(val)
        elif autofix and isinstance(val, str):
            return conv_str_auto(val)
        return val
    def autofix(self):
        for x, val in self.items():
            if isinstance(val, str):
                self[x] = conv_str_auto(val)
        return self

def get_args(args: str=None) -> MyDict:
    dict_ret = MyDict()
    if args is None:
        args = sys.argv
        dict_ret["__fname"] = args[0]
    else:
        args = re.sub("\s+", " ", args).split(" ")
    for i, x in enumerate(args):
        if   x[:4] == "----":
            # この引数の後にはLISTで格納する
            dict_ret[x[4:]] = []
            for _x in args[i+1:]:
                if _x[:2] != "--": dict_ret[x[4:]].append(_x)
                else: break
        elif x[:3] == "---":
            dict_ret[x[3:]] = True
        elif x[:2] == "--":
            dict_ret[x[2:]] = args[i+1]
    return dict_ret

def setattr_deep(target: object, path: str, value: object):
    if path.find(".") >= 0:
        paths = path.split(".")
        obj   = target
        for _path in paths[:-1]:
            obj = getattr(obj, _path)
        setattr(obj, paths[-1], value)
    else:
        setattr(target, path, value)
    
def check_type(instance: object, _type: Union[object, List[object]]):
    _type = [_type] if not (isinstance(_type, list) or isinstance(_type, tuple)) else _type
    is_check = [isinstance(instance, __type) for __type in _type]
    if sum(is_check) > 0:
        return True
    else:
        return False

def check_type_list(instances: List[object], _type: Union[object, List[object]], *args: Union[object, List[object]]):
    """
    Usage::
        >>> check_type_list([1,2,3,4], int)
        True
        >>> check_type_list([1,2,3,[4,5]], int, int)
        True
        >>> check_type_list([1,2,3,[4,5,6.0]], int, int)
        False
        >>> check_type_list([1,2,3,[4,5,6.0]], int, [int,float])
        True
    """
    if isinstance(instances, list) or isinstance(instances, tuple):
        for instance in instances:
            if len(args) > 0 and isinstance(instance, list):
                is_check = check_type_list(instance, *args)
            else:
                is_check = check_type(instance, _type)
            if is_check == False: return False
        return True
    else:
        return check_type(instances, _type)

def makedirs(dirpath: str, exist_ok: bool = False, remake: bool = False):
    dirpath = correct_dirpath(dirpath)
    if remake and os.path.isdir(dirpath): shutil.rmtree(dirpath)
    os.makedirs(dirpath, exist_ok = exist_ok)

def correct_dirpath(dirpath: str) -> str:
    if os.name == "nt":
        return dirpath if dirpath[-1] == "\\" else (dirpath + "\\")
    else:
        return dirpath if dirpath[-1] == "/" else (dirpath + "/")

def strfind(pattern: str, string: str, flags=0) -> bool:
    if len(re.findall(pattern, string, flags=flags)) > 0:
        return True
    else:
        return False

def conv_str_auto(string):
    if   strfind(r"^[0-9]+$", string) or strfind(r"^-[0-9]+$", string) or strfind(r"^[0-9]+\.0+$", string) or strfind(r"^-[0-9]+\.0+$", string): return int(float(string))
    elif strfind(r"^[0-9]+\.[0-9]+$", string) or strfind(r"^-[0-9]+\.[0-9]+$", string): return float(string)
    elif string in ["true",  "True"]:  return True
    elif string in ["false", "False"]: return False
    return string