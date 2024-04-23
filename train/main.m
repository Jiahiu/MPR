clear;
tic
for sn = 1
data1 = loadData(['C:\Users\jhc\Desktop\毕设\实验数据\S0',num2str(sn),'\uf']);
[feat1,class1] = getFeature(data1);
data2 = loadData(['C:\Users\jhc\Desktop\毕设\实验数据\S0',num2str(sn),'\fa']);
[feat2,class2] = getFeature(data2);
data = [feat1;feat2];
label = [class1;class2];
[xtrain,xtest,ytrain,ytest] = data_split(data,label);

model = svmModel();
model.train(xtrain,ytrain);
[P(sn),yp] = model.test(xtest,ytest);
end
toc
