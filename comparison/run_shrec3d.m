function run_shrec3d(input_path, output_path)
    folder_dir = "/rhome/yhu/bigdata/proj/experiment_G3DM/chromosome_3D/comparison/ShRec-Exented" 
    % Add that folder plus all subfolders to the path.
    addpath(genpath(folder_dir));
    DataF=importdata( fullfile(input_path, "norm_mat.txt") );
    XYZ = ShRec3D_ext(DataF, 1, 'sammon');
    % XYZ=ShRec3D(DataF);
    fig=figure,plot3(XYZ(:,1),XYZ(:,2),XYZ(:,3));
    saveas(fig,fullfile(output_path, 'conformation.jpg'));
    save(fullfile(output_path, 'conformation.xyz'), 'XYZ', '-ascii');
end

