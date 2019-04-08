import logging
import math

logger = logging.getLogger(__name__)


def estimate(img_paths, batch_size, epochs, override, train):
    logger.info(' ' + str(img_paths) + ' images')
    logger.info(' ' + str(batch_size) + ' images per batch')
    iterations_per_epoch = img_paths/batch_size
    if train:
        logger.info(' ~' + str(math.floor(iterations_per_epoch))+ ' iterations per epoch')
        it_total = iterations_per_epoch * epochs
        logger.info(' ' + str(epochs) + ' epochs, so ' + str(math.floor(it_total)) + ' iterations total')
    else:
        it_total = iterations_per_epoch
        logger.info(' ' + str(math.floor(it_total)) + ' iterations total')

    min_time = 3
    max_time = 7

    if not train:
        min_time = 5
        max_time = 10

    if 5*it_total > 3600:
        it_total_h = it_total/3600
        time = str(math.floor(min_time*it_total_h*10)/10) + '-' + str(math.floor(max_time*it_total_h*10)/10) + ' hours'
    elif 5*it_total > 60:
        it_total_m = (it_total/60)
        time = str(math.floor(min_time*it_total_m)) + '-' + str(math.floor(max_time*it_total_m)) + ' minutes'
    else:
        it_total = it_total
        time = str(math.floor(min_time*it_total)) + '-' + str(math.floor(max_time*it_total)) + ' seconds'

    logger.info(' Assuming ' + str(min_time) + '-' + str(max_time) + ' sec per iteration based on your cpu, that is ' + time)

    if override:
        logger.info(' Override: True')
        return True, it_total

    return confirm(prompt='Is that ok?', resp=True), it_total


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

