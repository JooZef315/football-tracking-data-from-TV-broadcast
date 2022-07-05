'''utils for argparse
'''


from utils import utils


def str2bool(v: str) -> bool:
    return v.lower() in ('true', '1', 'yes', 'y', 't')


def print_opt(opt):
    content_list = []
    args = list(vars(opt))
    args.sort()
    for arg in args:
        content_list += [arg.rjust(25, ' ') + '  ' + str(getattr(opt, arg))]
    utils.print_notification(content_list, 'OPTIONS')


def confirm_opt(opt):
    print_opt(opt)
    if not utils.confirm():
        exit(1)
