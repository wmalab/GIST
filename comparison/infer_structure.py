import sys, os
import subprocess

def run_command(command, path):
    print('cwd: {}\ncommand: {}'.format(path, command))
    p = subprocess.Popen(command, cwd=path, shell=True)
    p.wait()

def run(path, method, cell, resolution, chromosome):
    cwd_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
    if method=='pastis':
        input_path = '.'
        command = "pastis-pm2 {}".format(input_path)
    elif method=='shrec3d':
        input_path = '.'
        output_path = '.'
        command = "matlab -nodesktop -nodisplay -nosplash -r \'run_shrec3d(\"{}\", \"{}\"); quit;\'".format(input_path, output_path)
    elif method=='gem':
        input_path = '.'
        command = "matlab -nodesktop -nodisplay -nosplash -r \'run_GEM(\"{}\"); quit;\'".format(input_path)
    run_command(command, cwd_path)

if __name__ == '__main__':
    raw_hic_path = '/rhome/yhu/bigdata/proj/experiment_G3DM/data/raw'
    name = 'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool'
    chromosome = '21'
    cell = name.split('.')[0]
    resolution = name.split('.')[1]

    path = '/rhome/yhu/bigdata/proj/experiment_G3DM/'
    method = 'shrec3d'

    run(path, method, cell, resolution, chromosome)