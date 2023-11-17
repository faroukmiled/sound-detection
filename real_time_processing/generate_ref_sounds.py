import subprocess
import sys
import time
sys.path.append("/home/fmiled/compete")
for i in range(11,21):
    command = ['/bin/python3.9', 'record_sound.py', '--filename=./samples/ping'+str(i)+'.wav']
    process = subprocess.Popen(command)
    time.sleep(5)