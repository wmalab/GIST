import sys, os
import subprocess

def run_command(command, path):
    print('cwd: {}\ncommand: {}'.format(path, command))
    p = subprocess.Popen(command, cwd=path, shell=True)
    p.wait()

def run(path, method, cell, resolution, chromosome):
    # cwd_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
    cwd_path = os.path.join(path, 'chromosome_3D', 'comparison')
    if method=='pastis':
        input_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
        command = "pastis-pm2 {}".format(input_path)
    elif method=='shrec3d':
        input_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
        output_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
        command = "matlab -nodesktop -nodisplay -nosplash -r \'run_shrec3d(\"{}\", \"{}\"); quit;\'".format(input_path, output_path)
    elif method=='gem':
        input_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
        output_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
        command = "matlab -nodesktop -nodisplay -nosplash -r \'run_gem(\"{}\"); quit;\'".format(input_path, output_path)
    run_command(command, cwd_path)

if __name__ == '__main__':
    raw_hic_path = '/rhome/yhu/bigdata/proj/experiment_G3DM/data/raw'
    name = 'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool'
    chromosome = str(sys.argv[1]) #'21'
    cell = name.split('.')[0]
    resolution = name.split('.')[1]

    path = '/rhome/yhu/bigdata/proj/experiment_G3DM/'
    method = str(sys.argv[2]) #'gem'

    run(path, method, cell, resolution, chromosome)