function run_shrec3d(input_path, output_path)
    folder = "ShRec-Exented" 
    % Add that folder plus all subfolders to the path.
    addpath(genpath(folder));
    DataF=importdata( fullfile(input_path, "norm_mat.txt") );
    XYZ = ShRec3D_ext(DataF, 1, 'sammon')
    % XYZ=ShRec3D(DataF);
    fig=figure,plot3(XYZ(:,1),XYZ(:,2),XYZ(:,3));
    saveas(fig,strcat(output_path, '.jpg'));
    save(strcat(output_path, '.xyz'), 'XYZ', '-ascii');
end

