classdef svmModel < handle
    properties
        model_arg = [];
        w_arg = [];
    end 
    methods
        function  out = fsl_c(obj)
            w_cell = obj.w_arg;
            n = length(w_cell);
            nl = length(w_cell{1});
            c = zeros(nl,1);
            for i = 1:n
                c = c + w_cell{i};
            end
            out = c;
        end

        function obj = svmModel()
            obj.model_arg = 0;
        end

        function out = train(obj,trainData,trainClass,s)
            y = trainClass;
            x = trainData;
            if nargin>3
            x = trainData(:,s);
            end
            n = length(unique(y));
            trainData = cell(1,n*(n-1)/2);
            trainLabel = cell(1,n*(n-1)/2);
            svm_model = cell(1,n*(n-1)/2);
            w_cell = cell(1,n*(n-1)/2);
            index = cell(1,n);
            %--- 单独分开每个类
            X = cell(1,n);
            for i = 1:n
                index{i} = find(y==i);
                X{i} = x(index{i},:);
            end
            num = 0;
            %--- 训练
            for j = 1:n-1
                for i = j+1:n
                    num = num+1;
                    trainData{1,num}=[X{i};X{j}];
                    trainLabel{1,num}=[zeros(1,size(X{i},1)),ones(1,size(X{j},1))];
                    XX = trainData{1,num};
                    YY = trainLabel{1,num};
                    svmmodel=fitcsvm(XX,YY,'KernelScale','auto','Standardize',true,...
                        'OutlierFraction',0.05);
                    svm_model{1,num} = svmmodel;
                    w = svmmodel.Beta.^2;
                    w_cell{1,num} = w;
                end
            end
            obj.model_arg = svm_model;
            obj.w_arg = w_cell;
            out = [];
        end
%---------------------- test
        function [Pe,yp] = test(obj,testData,testClass,s)
            if nargin>3
            xt = testData(:,s);
            else
                xt = testData;
            end
            nclass = length(unique(testClass));
            yt = testClass;
            n = length(unique(yt));
            svm_model = obj.model_arg;
            for j=1:n*(n-1)/2
                y_pred = predict(svm_model{1,j},xt);  
                y_pred(y_pred>=0.5) = 1;
                y_pred(y_pred<0.5) = 0;
                result(:,j) = y_pred;  
            end

            nd = size(result,1);% 样本总数
            score = zeros(nd,n);
            for k=1:nd
                num = 0;
                for j = 1:n-1
                    for i = j+1:n
                        num = num+1;
                        if(result(k,num)==1)
                            score(k,j) = score(k,j)+1;
                        else
                            score(k,i) = score(k,i)+1;
                        end
                    end
                end
            end
            [~,yp] = max(score');
            %%%%
            Pe = length(find(yp==yt'))/size(yt,1);
            mtr = [];
            for i = 1:n
                idyt = find(yt==i);
                tmpm = sum(yp(idyt)==i)/length(idyt);
                mtr = [mtr,tmpm];
            end
            %-------tmp :分别计算fa-nl的acc
            % idyt = find(yt==i);
            % tmpm = sum(yp(idyt)==i)/length(idyt);
            % mtr = [mtr,tmpm];

             %----  计算精确度和召回率
            TP = [];
            TN = [];
            FP = [];
            FN = [];
            TestPredict = yp;
            for i = 1:nclass
                index = testClass==i;
                index_ = testClass~=i;
                TP(i) = sum(TestPredict(index)==i)/sum(index);
                TN(i) = sum(TestPredict(index)~=i)/sum(index_);
                FP(i) = sum(TestPredict(index)==i)/sum(index_);
                FN(i) = sum(TestPredict(index)~=i)/sum(index);
            end
            metrics = [TP;TN;FP;FN];
            Precision_ = TP/(TP+FP);
            Recall_ = TP/(TP+FN);
            F1 = 2*Recall_*Precision_/(Recall_+Precision_);  
            yp = yp';
        end
    end
end