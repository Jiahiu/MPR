
emg = td(8000:48000,2);
% 创建一个图形窗口，设置分辨率
figure('Renderer', 'painters', 'Position', [100 100 800 600]);

% 绘制原始信号
subplot(2,1,1); % 两行一列的第一个
plot(emg, 'LineWidth', 1);
title('Raw EMG Signal');
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
set(gca, 'FontSize', 10, 'FontName', 'Times New Roman'); % 设置字体和大小

emg_filtered = td2(8000:48000,2);
% 绘制滤波后的信号
subplot(2,1,2); % 两行一列的第二个
plot(emg_filtered, 'LineWidth', 1);
title('Filtered EMG Signal');
xlabel('Time (ms)');
ylabel('Amplitude (mV)');
set(gca, 'FontSize', 10, 'FontName', 'Times New Roman'); % 设置字体和大小

% 设置整个图形的字体和大小
set(findall(gcf,'-property','FontSize'),'FontSize',10)
set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')

% 保存图像为矢量图，以便在放大时保持清晰
print(gcf, 'emg_signal_plot', '-depsc');