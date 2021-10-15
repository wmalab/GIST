import sys
import subprocess

def run_pastis(path):
    command = "pastis-pm2 ."
    p = subprocess.Popen(command, cwd=path, shell=True)
    p.wait()
if __name__ == '__main__':
    pass