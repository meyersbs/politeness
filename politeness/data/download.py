import os
import progressbar as pb
import sys

from datetime import datetime as dt
from urllib.request import urlretrieve

#### PACKAGE IMPORTS ###########################################################
from politeness.helpers import get_elapsed

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
WIKI_JSON = THIS_DIR + "/wikipedia.parsed.json"
WIKI_URL = "http://people.rc.rit.edu/~bsm9339/corpora/stanford_politeness/wikipedia.parsed.json"
STACK_JSON = THIS_DIR + "/stack-exchange.parsed.json"
STACK_URL = "http://people.rc.rit.edu/~bsm9339/corpora/stanford_politeness/stack-exchange.parsed.json"

def download_progress(blocks_read, block_size, total_size):
    if blocks_read == 0:
        widgets = [pb.AnimatedMarker(markers='←↖↑↗→↘↓↙'), ' ', pb.Percentage(),
                   ' ', pb.Bar(), ' ', pb.ETA(), ' ', pb.FileTransferSpeed()]
        download_progress.progress_bar = pb.ProgressBar(widgets=widgets,
            maxval=(total_size + block_size)).start()

    download_progress.progress_bar.update(blocks_read * block_size)

def _download(out_path, url):
    (_, headers) = urlretrieve(url, out_path, reporthook=download_progress)
    download_progress.progress_bar.finish()

def download():
    start = dt.now()
    print("({:s}) Downloading File 1/2: wikipedia.parsed.json".format(str(start)[11:19]))
    _download(WIKI_JSON, WIKI_URL)
    print("    ({:s}) Finished!".format(str(dt.now())[11:19]))
    print('    Time: {:.2f} minutes.'.format(get_elapsed(start, dt.now())))
    print('\n')

    start = dt.now()
    print("({:s}) Downloading File 2/2: stack-exchange.parsed.json".format(str(start)[11:19]))
    _download(STACK_JSON, STACK_URL)
    print("    ({:s}) Finished!".format(str(dt.now())[11:19]))
    print('    Time: {:.2f} minutes.'.format(get_elapsed(start, dt.now())))
    print('\n')

def __reset():
    open(WIKI_JSON, 'w').close()
    open(STACK_JSON, 'w').close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        download()
    elif args[0] == 'download':
        download()
    elif args[0] == 'reset':
        __reset()
    else:
        print("Received invalid command.")
