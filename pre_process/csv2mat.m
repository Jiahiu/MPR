% clear;clc
% subject_num = 3;
for tn = 1:2
loadPath = ['C:\Users\jhc\Desktop\毕设\实验数据\S0',num2str(subject_num),'\t',num2str(tn),'\'];
files = dir([loadPath,'*.csv']);
csv2matf(loadPath,files,tn);
end
xi_cell1 = cell(1,8);
xi_cell2 = cell(1,8);
save('tmp//xi_cell1.mat',"xi_cell1")
save('tmp//xi_cell2.mat',"xi_cell2")

function csv2matf(loadPath,files,tn)
k = 1:12;
ix = 8*(k-1)+2;
data = cell(1,8);
for i = 1:length(files)/2
    name1 = [loadPath,files(i).name];
    name2 = [loadPath,files(i+8).name];
    data1 = readmatrix(name1);
    data1 = data1(:,ix);
    data2 = readmatrix(name2);
    data2 = data2(:,ix);
    tmpdata = [data1;data2];
    data{i} = tmpdata; 
end
saveName = ['tmp//t',num2str(tn),'_data.mat'];
save(saveName,'data');
end