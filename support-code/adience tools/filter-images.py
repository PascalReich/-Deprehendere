import os
import re
import logging
import argparse

items_to_remove = []

def purge(dir, pattern):
    for f in os.listdir(dir):
        try:
            for j in os.listdir(os.path.join(dir,f)):
                if re.search(pattern, j):
                    items_to_remove.append(os.path.join(dir, f, j))
        except NotADirectoryError:
            print("skipping a file")

    print(len(items_to_remove))
    confirm()

    for i in items_to_remove:
        os.remove(i)

def confirm(prompt=None, resp=False):
    """prompts for yes or no response from the user. Returns True for yes and
    False for no.

    'resp' should be set to the default value assumed by the caller when
    user simply types ENTER.

    >>> confirm(prompt='Create Directory?', resp=True)
    Create Directory? [y]|n:
    True
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y:
    False
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: y
    True

    """

    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')

    while True:
        ans = input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print('please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--dir-path', type=str, action='store', dest='dir_path',
                        help='Path to model protobuf graph')
    parser.add_argument('--regex', type=str, action='store', dest='regex',
                        help='Input path of data to train on')

    args = parser.parse_args()

    purge(dir=args.dir_path,pattern=args.regex)
