function run_gem(input_path, output_path)
    % Determine where your m-file's folder is.
    folder_dir = "/rhome/yhu/bigdata/proj/experiment_G3DM/chromosome_3D/comparison/GEM"
    % Add that folder plus all subfolders to the path.
    addpath(genpath(folder_dir));
    x = fullfile( input_path, "norm_mat.txt")
    bin = fullfile( input_path, "loci.txt")
    GEM(x, bin, 1E3, 10, 5E12, 0, -1, output_path);
end