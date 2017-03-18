from easypy.colors import Colorized
from easypy.collections import listify


class CancelledException(KeyboardInterrupt):
    pass


def message(fmt, *args, wait_for_user=False, **kwargs):
    msg = fmt.format(*args, **kwargs) if (args or kwargs) else fmt
    print(Colorized(msg))
    if wait_for_user:
        globals()['wait_for_user']()


def get_input(prompt, default=NotImplemented):
    prompt = prompt.strip(": ")
    if default is not NotImplemented:
        prompt = "%s DARK_CYAN<<[%s]>>" % (prompt, default)
    ret = input(Colorized(prompt + ": "))
    if not ret:
        if default is NotImplemented:
            raise CancelledException()
        else:
            return default
    return ret


def wait_for_user():
    try:
        get_input("(Hit <enter> to continue)")
    except CancelledException:
        pass


def choose(question, options, default=NotImplemented):
    options = list(options.items() if isinstance(options, dict) else options)

    all_options = {}
    for opt, value in options:
        for o in listify(opt):
            all_options[o.lower()] = value

    disp_options = "/".join(str(opt[0] if isinstance(opt, (list, tuple)) else opt) for opt, value in options)
    fmt = "%s (%s) " if default is NotImplemented else "%s [%s]"
    msg = fmt % (question, disp_options)

    while True:
        answer = get_input(msg, default=default).strip().lower()
        if answer not in all_options:
            print("Invalid answer ('%s')" % answer)
            continue
        return all_options[answer]


def ask(question, default=NotImplemented):
    if default is NotImplemented:
        options = [(("y", "yes"), True), (("n", "no"), False)]  # no default
    elif default:
        default = "Y"
        options = [(("Y", "yes"), True), (("n", "no"), False)]  # 'Y' is default
    else:
        default = "N"
        options = [(("y", "yes"), True), (("N", "no"), False)]  # 'N' is default

    try:
        return choose(question, options, default=default)
    except EOFError:
        return False  # EOF == no
