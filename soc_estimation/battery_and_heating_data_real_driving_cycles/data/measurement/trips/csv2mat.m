clear; clc; close all;

files = dir('*.csv');

for i = 1:length(files)
    csvFileName = files(i).name;
    data = readtable(csvFileName);
    [~, name, ~] = fileparts(csvFileName);
    matFileName = strcat(name, '.mat');
    
    save(matFileName, 'data');
end

