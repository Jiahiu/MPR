function data = loadData(path)
files = dir([path,'\*.mat']);
len = length(files);
data = cell(1,len);
for k = 1:len
    cleanName = files(k).name;
    cleanData = importdata([path,'\',cleanName]);
    data{1,k}=cleanData;
end
