function [xtrain,xtest,ytrain,ytest] = data_split(data,label)
rdn = randperm(length(label));
data = data(rdn,:);
label = label(rdn);
num = floor(length(label)*0.7);
xtrain= data(1:num,:);
ytrain = label(1:num);
xtest = data(num+1:end,:);
ytest = label(num+1:end);
end