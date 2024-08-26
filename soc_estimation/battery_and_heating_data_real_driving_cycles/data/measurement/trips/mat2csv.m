clear; clc; close all;

files = dir('*.mat');

for i = 1:length(files)
    data = load(files(i).name);
    [~, fileName, ~] = fileparts(files(i).name);
    csv_filename = [fileName '.csv'];
    
    table = array2table(data);
    
    writetable(table.data.data, csv_filename, 'WriteVariableNames',true);
end
