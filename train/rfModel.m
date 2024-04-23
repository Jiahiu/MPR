classdef rfModel < handle
    properties
        model_arg = [];
        model_numTrees = 0;
    end
    
    methods
        function obj = rfModel()
            obj.model_arg = 0;
        end

        function out = train(obj,trainData,trainClass)
            parms = [22,6,96];
            ytrain = trainClass;
            xtrain = trainData;
            numTrees = parms(1);
            numFeatures = parms(2);
            treeDepth = parms(3);
            rng(42)
            n = length(unique(ytrain));
            %--------训练
            forest = cell(numTrees, 1); 
            for i = 1:numTrees
                % 随机抽样
                indices = randperm(size(xtrain, 1)*0.9);
                sampled_X_train = xtrain(indices, :);
                sampled_y_train = ytrain(indices);
                tree = fitctree(sampled_X_train, sampled_y_train, ...
                    'MaxNumSplits', treeDepth, 'NumVariablesToSample', numFeatures);
                
                forest{i} = tree;
            end
            obj.model_arg = forest;
            obj.model_numTrees = numTrees;
            out = [];
        end

        function [acc,yp] = test(obj,testData,testClass)
            xtest = testData;
            ytest = testClass;
            forest = obj.model_arg;
            n = length(unique(ytest));
            numTestSamples = size(xtest, 1);
            numTrees = obj.model_numTrees;
            predictions = zeros(numTestSamples, numTrees);
            for i = 1:numTrees
                tree = forest{i};
                predictions(:, i) = predict(tree, xtest);
            end
            yp = mode(predictions, 2);
            
            acc = sum(yp == ytest) / numel(ytest);
            macc = [];
            for i = 1:n
                idyt = find(ytest==i);
                tmpm = sum(yp(idyt)==i)/length(idyt);
                macc = [macc,tmpm];
            end

            nclass = n;
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
        end
    end
end