import sys
import subprocess

def run_command(command, path):
    p = subprocess.Popen(command, cwd=path, shell=True)
    p.wait()

def run(method, input_path, output_path):
    if method=='pastis':
        command = "pastis-pm2 ."
        path = ''
    elif method=='shrec3d':
        command = "matlab -nodesktop -nodisplay -nosplash -r 'run_shrec3d(\"{}\", \"{}\"); quit;'".format(input_path, output_path)
    elif method=='gem':
        command = "matlab -nodesktop -nodisplay -nosplash -r 'run_GEM({}, {}); quit;".format(input_path, output_path)
        pass
    
    run_command(command, path)

if __name__ == '__main__':
    pass