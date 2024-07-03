import torch
import gc

from colorama import Fore, Style


# Getting objects and Cleaning memory

def get_all_to_return(to_return, fillwith=None) -> list:
    if isinstance(to_return, str):
        to_return = [to_return]
    vars_to_return = []
    available = []

    for var_str in to_return:
        try:
            vars_to_return.append(get_var_from_name(var_str))
        except:
            vars_to_return.append(fillwith)
    if len(vars_to_return)>=1:
        if len(to_return)>len(vars_to_return):
            print(f"Only: {available} were available. Lacking filled with {fillwith}")
        print(f"Available: {len(vars_to_return)}")
        return vars_to_return
    else:
        print('None of the elements were found, returning empty list')
        return []

def get_var_from_name(passed):
    # if string, return object
    if isinstance(passed, str):
        if passed in globals():
            print(f'{passed} in globals')
            return globals()[passed]
        elif passed in locals():
            print(f'{passed} in locals')
            return locals()[passed]
        else:
            print(f'{passed} not in globals() and locals()')
    else:
        raise TypeError(f'{passed} is not a string')


def del_obj(el):
    try:
        if isinstance(el, str):
            eval(f"del {el}")
        else:
            del el
    except:
        print(Fore.RED + f"Warning: {el} cannot be deleted by clean_memory(), skipping..." + Style.RESET_ALL)

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# from colorama import Fore, Style
#
# def get_all_to_return(to_return, fillwith=None) -> list:
#     if isinstance(to_return, str):
#         to_return = [to_return]
#     vars_to_return = []
#     available = []
#
#     for var_str in to_return:
#         try:
#             vars_to_return.append(get_var_from_name(var_str))
#         except:
#             vars_to_return.append(fillwith)
#     if len(vars_to_return)>=1:
#         if len(to_return)>len(vars_to_return):
#             print(f"Only: {available} were available. Lacking filled with {fillwith}")
#         print(f"Available: {len(vars_to_return)}")
#         return vars_to_return
#     else:
#         print('None of the elements were found, returning empty list')
#         return []
#
# def get_var_from_name(passed, locs, globs):
#     # if string, return object
#     if isinstance(passed, str):
#         if passed in globs:
#             return globs[passed]
#         elif passed in locs:
#             return locs[passed]
#         else:
#             print(f'{passed} not in globals() and locals()')
#     else:
#         raise TypeError(f'{passed} is not a string')
#
#
# def del_obj(el, locs, globs):
#     try:
#         if isinstance(el, str):
#             el_to_del = get_var_from_name(el, locs, globs)
#         else:
#             el_to_del = el
#         del el_to_del
#         print(f'{el} deleted')
#     except:
#         print(Fore.RED + f"Warning: {el} cannot be deleted by clean_memory(), skipping..." + Style.RESET_ALL)
#
#
# def clean_memory(to_del, locs, globs):
#     if to_del:
#         if not isinstance(to_del, list):
#             to_del = [to_del]
#         for el in to_del:
#             del_obj(el, locs, globs)
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats()