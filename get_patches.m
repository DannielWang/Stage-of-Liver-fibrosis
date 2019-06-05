%% Create new folders
folders = {'../DCM', '../HCM', '../Normal'};
patch_labels = {'D', 'H', 'N'};
output_patch_folder = './patches_20x';
for i = 1:length(folders)
    if(~exist(fullfile(folders{i},output_patch_folder), 'dir'))
        mkdir(fullfile(folders{i},output_patch_folder));
    end
end

%% Get patches
patch_size = [144, 144];

for i=1:length(folders)
    files = dir( fullfile(folders{i}, '20x', '*.tif') );
    for j=1:length(files)
        filename = fullfile(files(j).folder, files(j).name);
        [pathstr, name, ext] = fileparts(files(j).name);
        fprintf('generating patches for %s... ', filename); timerID = tic; 
        img = imread( filename );
        patch_num = floor([size(img,1) size(img,2)]./patch_size);
        img_cropped = img(1:patch_size(1) * patch_num(1), 1:patch_size(2) * patch_num(2), :);
        patches = mat2cell(img_cropped, ones(1,patch_num(1))*patch_size(1), ones(1,patch_num(2))*patch_size(2), 3);
        for k=1:patch_num(1)
            for l=1:patch_num(2)
                imwrite(patches{k,l}, fullfile(folders{i},output_patch_folder, sprintf('%s_%s_%02d_%02d.png', patch_labels{i}, name, k, l)));
            end
        end
        fprintf('done in %f sec\n', toc(timerID));
    end
end

% img = imread('C:\Users\icb\Desktop\test_t_SNE\test\DCM\20x\1950HE20-3.tif');
% patch_num = floor([size(img,1) size(img,2)]./patch_size);