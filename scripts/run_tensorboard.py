import os
import signal
import subprocess
import requests
import json
import time
BOLD = '\033[1m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
ENDC = '\033[0m'
RLTRADE_PREFIX_PATH = '/mnt/pollux-new/cis/kearnsgroup/kearnsgroup/RLtrade/'

def initialize_processes():
    processes = []
    FNULL = open(os.devnull, 'w')

    print 'Starting tensorboard...'
    TENSORBOARD_COMMAND = [
        'python',
        RLTRADE_PREFIX_PATH + 'RL/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py',
        '--logdir=' + RLTRADE_PREFIX_PATH + 'tensorboard/tensorboard-logging-output'
    ]
    processes.append(subprocess.Popen(TENSORBOARD_COMMAND, stdout=FNULL, preexec_fn=os.setsid))
    time.sleep(6)
    print 'Tensorboard initialized.'

    print 'Initializing ngrok...'
    TENSORBOARD_PORT = 6006
    NGROK_COMMAND = [
        RLTRADE_PREFIX_PATH + 'tensorboard/./ngrok',
        'http',
        str(TENSORBOARD_PORT)
    ]
    processes.append(subprocess.Popen(NGROK_COMMAND, stdout=FNULL, preexec_fn=os.setsid))
    print 'ngrok started.'

    return processes


def get_tensorboard_url():
    print 'Querying tunnel path...'
    url = "http://localhost:4040/api/tunnels"
    headers = {
        'cache-control': "no-cache",
    }
    response = requests.request("GET", url, headers=headers)
    return response.json()['tunnels'][0]['public_url']


def kill_procs(processes):
    for proc in processes:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)


def handle_kill(signal, frame):
    return


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_kill)

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    print 'Initializing processes...'
    processes = initialize_processes()
    time.sleep(3)

    tunnel_path = get_tensorboard_url()
    print BOLD + OKBLUE + "Access Tensorboard on your local machine at:"
    print OKGREEN + tunnel_path + ENDC

    command = ''
    while(True):
        command = raw_input("Type 'quit' to exit.\n")

        if command.lower() == 'quit':
            kill_procs(processes)
            exit(0)
