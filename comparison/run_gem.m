function run_GEM(input_file, output_file)
    % Determine where your m-file's folder is.
    folder = "GEM"
    % Add that folder plus all subfolders to the path.
    addpath(genpath(folder));
    x = importdata( fullfile( input_file, "norm_mat.txt") );
    bin = importdata( fullfile( input_file, "loci.txt") )
    GEM(x, bin, 1E4, 10, 5E12, 0, -1, output_file);
end