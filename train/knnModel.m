classdef knnModel < handle
    properties
        model = [];
    end
    methods
        function obj = knnModel()
            obj.model = [];
        end
        
        function y = train(obj,xtrain,ytrain)
            K = 3;
            Mdl = fitcknn(xtrain, ytrain, 'NumNeighbors', K);
            obj.model = Mdl;
            y = [];
        end
        
        function [acc,ypred ] = test(obj,xtest,ytest)
            Mdl = obj.model;
            ypred = predict(Mdl, xtest);
            acc = sum(ypred == ytest) / length(ytest);
            nclass = length(unique(ytest));
             %----  计算精确度和召回率
            TP = [];
            TN = [];
            FP = [];
            FN = [];
            testClass = ytest;
            TestPredict = ypred;
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
        end
    end
end
