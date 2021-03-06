function run_chromsde(input_path, chrnum, method_type, output_path)
    % Determine where your m-file's folder is.
    folder_dir = "/rhome/yhu/bigdata/proj/experiment_GIST/chromosome_3D/comparison/ChromSDE/program"
    addpath(genpath(folder_dir));
    % Add that folder plus all subfolders to the path.
    % Input data: (assume n loci)
    % bin: nx4 matrix, the description for each 3d point (id, chromosome, start, end)
    % x: nxn sparse matrix,  the normalized contact frequency matrix
    % method_type: 1 for quadratic SDP and 0 for linear SDP

    x = fullfile( input_path, "norm_mat.txt")
    trainFreq = load(x);
    s=size(trainFreq, 2);
    cbin = fullfile( input_path, "loci.txt")
    trainBin = load(cbin);
    % trainBin=zeros(s,4);
    % chr = chrnum;
    % if (strcmp(chrnum, 'X'))
    %     chr = 23;
    % else
    %     chr = str2num(chrnum);
    % end
    % for i=1:s
    %     trainBin(i,2)=chr;
    %     trainBin(i,3)=1+(i-1)*resolution;
    %     trainBin(i,4)=i*resolution;
    %     trainBin(i,1)=i;
    % end
    size(trainBin)
    size(trainFreq)
    output_path
    ChromSDE(trainBin, trainFreq, method_type, output_path);
end
    