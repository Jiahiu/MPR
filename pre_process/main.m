% 1--加载数据
clear;clc;
tic
subject_num = 1;

% s
tn = 2;
load(['tmp//t',num2str(tn),'_data.mat'])
if tn==1
savePath = ['C:\Users\jhc\Desktop\毕设\实验数据\S0',num2str(subject_num),'\uf\'];
elseif tn==2
savePath = ['C:\Users\jhc\Desktop\毕设\实验数据\S0',num2str(subject_num),'\fa\']; 
end

% 2--滤波
n = length(data);
result = cell(1,n);
for i = 1:n
    td = data{i};
    td2 = bdfilter(td);
    % td3 = dwtfilter(td2);
    td4 = peakfilter(td2);
    result{i} = td4;
end

% 3--动作截取
result2 = cell(1,n);
for k = 5:n
    td = result{k};
    xi = getpt_my(td,k);
    acd = [];
    if(size(xi,1)==8)
        x1 = xi(:,1);x2 = xi(:,2);
        for i = 1:length(x1)
           cld = td(x1(i):x2(i),:);
           acd = [acd;cld];
        end
        name=['s',num2str(subject_num),'_class',num2str(k),'.mat'];
        save([savePath,name],'acd');
    elseif(size(xi,1)==4)
        x1 = xi(:,1);x2 = xi(:,2);
        for i = 1:length(x1)
           cld = td(x1(i):x2(i),:);
           acd = [acd;cld];
        end
        acd = [acd;acd];
        name=['s',num2str(subject_num),'_class',num2str(k),'.mat'];
        save([savePath,name],'acd');
    end
end
toc