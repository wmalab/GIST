import sys, os
import subprocess
import numpy as np
def format_chromsde(output_path):
    files = [f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))]
    for f in files:
        if '.pos' in f: pdb = f

    structure = []
    with open(os.path.join(output_path, pdb)) as file:
        next(file)
        for line in file:
            l = line.split(',')
            indx = int(float(l[0]))%10e10
            x,y,z = l[1:]
            structure.append([float(x),float(y),float(z)])
        file.close()
    structure = np.array(structure, ndmin=2)
    output_file = os.path.join(output_path, 'conformation.xyz')
    np.savetxt(output_file, structure, fmt="%f\t%f\t%f")

def format_lordg(output_path):
    files = [f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))]
    for f in files:
        if '.pdb' in f: pdb = f
        if 'mapping' in f: idx = f

    structure = list()
    with open(os.path.join(output_path, pdb)) as file:
        next(file)
        for line in file:
            if 'ATOM' in line:
                l = line.split()
                indx = l[1]
                x,y,z = l[5:8]
                structure.append([float(x),float(y),float(z),int(indx)])
            if 'CONECT' in line:
                break
        file.close()
    structure = np.array(structure, ndmin=2)

    indx = np.loadtxt( os.path.join(output_path, idx) )
    raw_indx = indx[:,0].astype(int)
    structure[ indx[:, 1].astype(int), -1] = raw_indx

    output_file = os.path.join(output_path, 'conformation.xyz')
    np.savetxt(output_file, structure, fmt="%f\t%f\t%f\t%d")
    return 

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
        os.makedirs(output_path, exist_ok=True)
        command = "matlab -nodesktop -nodisplay -nosplash -r \'run_shrec3d(\"{}\", \"{}\"); quit;\'".format(input_path, output_path)
    elif method=='gem':
        input_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
        output_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
        os.makedirs(output_path, exist_ok=True)
        command = "matlab -nodesktop -nodisplay -nosplash -r \'run_gem(\"{}\", \"{}\"); quit;\'".format(input_path, output_path)
    elif method=='lordg':
        jar_path = os.path.join(cwd_path, 'LorDG', 'bin')
        input_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome, 'config.ini')
        command = "java -jar {}/3DDistanceBaseLorentz.jar {}".format(jar_path, input_path)
    elif method=='chromsde':
        input_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
        output_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
        os.makedirs(output_path, exist_ok=True)
        command = "matlab -nodesktop -nodisplay -nosplash -r \'run_chromsde(\"{}\", \"{}\", 1, \"{}\"); quit;\'".format(input_path, chromosome, os.path.join(output_path,'out') )
    
    run_command(command, cwd_path)

    if method=='lordg':
        output_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome, 'output')
        format_lordg(output_path)
    
    if method =='chromsde':
        output_path = os.path.join(path, 'comparison', method, cell, resolution, chromosome)
        format_chromsde(output_path)

if __name__ == '__main__':
    raw_hic_path = '/rhome/yhu/bigdata/proj/experiment_GIST/data/raw'
    name = 'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool'
    chromosome = str(sys.argv[1]) #'21'
    cell = name.split('.')[0]
    resolution = name.split('.')[1]

    path = '/rhome/yhu/bigdata/proj/experiment_GIST/'
    method = str(sys.argv[2]) #'gem'

    run(path, method, cell, resolution, chromosome)