import glob
import os


def change_font_color(color='', text=None):
    if color == "black":
        return_str = "\033[30m"
    elif color == "red":
        return_str = "\033[31m"
    elif color == "green":
        return_str = "\033[32m"
    elif color == "yellow":
        return_str = "\033[33m"
    elif color == "blue":
        return_str = "\033[34m"
    elif color == "magenta":
        return_str = "\033[35m"
    elif color == "cyan":
        return_str = "\033[36m"
    elif color == "white":
        return_str = "\033[37m"
    elif color == "bright black":
        return_str = "\033[90m"
    elif color == "bright red":
        return_str = "\033[91m"
    elif color == "bright green":
        return_str = "\033[92m"
    elif color == "bright yellow":
        return_str = "\033[93m"
    elif color == "bright blue":
        return_str = "\033[94m"
    elif color == "bright magenta":
        return_str = "\033[95m"
    elif color == "bright cyan":
        return_str = "\033[96m"
    elif color == "bright white":
        return_str = "\033[97m"
    else:
        return_str = "\033[0m"
    if text:
        return_str += text + "\033[0m"
    return return_str


def raise_error(message):
    if message.find('E: ')==-1:
        message = "\nE: " + message
    raise Exception(change_font_color("bright red", message))


def raise_issue(message):
    if message.find('I: ')==-1:
        message = "\nI: " + message
    print(change_font_color("bright yellow", message))


def second_to_dhms_string(second, second_round=True):
    d, left = divmod(second, 86400)
    h, left = divmod(left, 3600)
    m, s = divmod(left, 60)
    str = ""
    d, h, m, s = int(d), int(h), int(m), int(s) if second_round else int(s) + second - int(second)
    if d!=0:
        str+=f"{d:d}d "
    if h!=0 or d!=0:
        str+=f"{h:02d}h "
    if m!=0 or h!=0 or d!=0:
        str+=f"{m:02d}m "
    str += f"{s:02d}s" if second_round else f"{s:2.2f}s"
    return str


def load_file_path_list(file_paths: str, extension=''):
    extension = extension.strip('*.')
    extension_str = f'/*.{extension}' if extension != '' else '/*.*'

    if os.path.isfile(file_paths):
        if (os.path.splitext(file_paths)[1] != '.' + extension) and extension != '':
            raise_error(f'Check the extension ! -> {file_paths}')
        return [os.path.normpath(os.path.abspath(file_paths))]
    return list(map(lambda file_path: os.path.normpath(os.path.abspath(file_path)), sorted(glob.glob(file_paths + '/**' + extension_str, recursive=True))))