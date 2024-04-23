function y=dwtfilter(data)

% 小波分解参数
wavelet_name = 'db4'; % 选择小波基
level = 5; % 分解层数

% 小波分解
y = [];
iend = size(data,2);
for i = 1:iend
m = data(:,i);
[c, l] = wavedec(m, level, wavelet_name);

% 选择阈值（可根据实际情况调整）
threshold = 1e-2 ;

% 阈值处理
% 阈值处理
cn = [];
a = appcoef(c, l, wavelet_name, level);
a_thresh = wthresh(a, 'h', threshold);
cn = [cn;a_thresh];
N = length(m);
for i = 1:level
    % 提取逼近系数和细节系数
    d = detcoef(c, l, level-i+1);
    
    % 阈值选择
    mad = median(abs(d));
    sigma = mad/0.6745;
    lamda = sigma*sqrt(2*log(N))/i;
    % 阈值处理
    d_thresh = wthresh(d, 'h', lamda);
    
    % 重构信号
    cn = [cn; d_thresh];
end


% 重构信号
td = waverec(cn, l, wavelet_name);
y = [y td];
% 绘制原始信号与滤波后的信号对比
% Fs = 1926; % 采样频率
% t = (0:length(m)-1) / Fs; % 时间向量
% figure;
% subplot(2,1,1);
% plot(t, m);
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Original EMG Signal');
% 
% subplot(2,1,2);
% plot(t, td);
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Filtered EMG Signal');
% p=1;
end
