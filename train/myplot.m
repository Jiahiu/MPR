classdef myplot
    methods
        function confumat(ypred,ytrue)
            y_pred = double(ypred);% ... (真实的类标签数据)
            y_true = double(ytrue);% ... (预测的类标签数据)
            
            % 使用confusionmat函数得到混淆矩阵
            confMat = confusionmat(y_true, y_pred);
            
            
            % 手动创建混淆矩阵图
            figure;
            imagesc(confMat); % 显示混淆矩阵
            colormap('parula');  % 选择一个内置colormap，尽量接近你所需的颜色
            colorbar;         % 显示颜色条
            
            % 添加轴标签
            xlabel('Predicted Label');
            ylabel('True Label');
            title('RF');
            
            % 添加数值标签
            numClasses = size(confMat, 1);
            for i = 1:numClasses
                for j = 1:numClasses
                    text(j, i, num2str(confMat(i,j),'%d'),...
                         'HorizontalAlignment', 'center',...
                         'Color', 'white');
                end
            end
            
            % 设定字体大小
            set(gca, 'FontSize', 12);
        end
    end
end