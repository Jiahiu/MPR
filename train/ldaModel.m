classdef ldaModel < handle
    properties
        Cg;
        Wg;       
    end
    methods
        function obj = ldaModel( )
            obj.Cg = [];
            obj.Wg = [];
        end
        %--  train
        function y = train(obj,trainData,trainClass)
            N = size(trainData,2); 
            nclass = length(unique(trainClass));
            y = 1:nclass;
            
            C = zeros(N,N); 
            for l = 1:nclass
	            idx = find(trainClass==y(l));			
	            Mi(l,:) = mean(trainData(idx,:));   
	            C = C + cov(trainData(idx,:)-ones(length(idx),1)*Mi(l,:));
            end
            
            C = C./nclass;
            Pphi = 1/nclass; 
            Cinv = inv(C);
            for i = 1:nclass
	            Wgg(:,i) = Cinv*Mi(i,:)';
	            Cgg(i) = -1/2*Mi(i,:)*Cinv*Mi(i,:)' + log(Pphi)';
            end
            obj.Cg = Cgg;
            obj.Wg = Wgg;
            y = [];
        end
        %--- test
        function [PeTest,yp] = test(obj,testData,testClass)
            Ptest = size(testData,1);
            Ate = testData*obj.Wg + ones(Ptest,1)*obj.Cg;
            AAte = compet(Ate');   
            errte = sum(sum(abs(AAte-ind2vec(testClass'))))/2;
            nete = errte/Ptest;     
            PeTest = 1-nete;
            TestPredict = vec2ind(AAte); %  预测结果
            nclass = length(unique(testClass));
            matr = [];
            for i = 1:nclass
                index = testClass==i;
                tt = sum(index);
                index2 = TestPredict(index)==i;
                ti = sum(index2);
                matr = [matr ti/tt];
            end
            
            yp = TestPredict';
             %----  计算精确度和召回率
            TP = [];
            TN = [];
            FP = [];
            FN = [];
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