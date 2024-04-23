    % var=FeatsFun(x,2);
    % log=FeatsFun(x,3);
    % vorder=FeatsFun(x,4);
    % psr=FeatsFun(x,8);
    % ttp=FeatsFun(x,9);
    % pkf=FeatsFun(x,10); 
    % iEMG=FeatsFun(x,13);
    % len = FeatsFun(x,14)/frameLen;
    % sumroot =FeatsFun(x,15);
    % meanexp = FeatsFun(x,16);
    % sis = FeatsFun(x,17);
    % rsd1 = FeatsFun(x,18);
    % rsd2 = FeatsFun(x,19);
    % zeroc = FeatsFun(x,20)/frameLen;
    % ssc = FeatsFun(x,21,thred)/frameLen;
    % wamp = FeatsFun(x,23,thred2)/frameLen;
    % entropyR = FeatsFun(x,24);

function y = FeatsFun(x,sn,args)
if sn==1
    y=RMS(x);
elseif sn==2
    y=varr(x);
elseif sn==3
    y=LogDetector(x);
elseif sn==4
    y=vorder(x);
elseif sn==6
    y=TDPSD(x);
elseif sn==7
    y=MeanPowerFrequency(x);
elseif sn==8
    y= PowerSpectralRatio(x);
elseif sn==9
    y = TotalPower(x);
elseif sn==10
    y= PeakFrequency(x);
elseif sn==11
     y = MedianFrequency(x);
elseif sn==12
     y=meandf(x); 
elseif sn==13
    y=iEMG(x);
elseif sn==14
    y=wavelen(x);
elseif sn==15
    y=sumRoot(x);
elseif sn==16
    y=meanExp(x);
elseif sn==17
    y=sis(x);
elseif sn==18
    y=rsd1(x);
elseif sn==19
    y=rsd2(x);
elseif sn==20
    y=zerocrossing(x);
elseif sn==21
    y=ssc(x,args);  %1/1e6*7
elseif sn==22
    y=mav(x);
elseif sn==23
    y=wamp(x,args);
elseif sn==24
    y=myentropy(x);
elseif sn==25
    y=ar_coef(x,4);
elseif sn==26
    y = sampleEntropy(x, 2, 0.25*std(x));
elseif sn==27
    y= dwtEnergy(x);
elseif sn==28
    y=mif(x);
elseif sn==29
    y=fi5(x);
end
end

%----5阶谱矩----fi5----s26
function y=fi5(x)
[f,P] = emgDFT(x);
f = f(2:end);
P = P(2:end);
fi5 = sum(f.^(-1).*P/sum(f.^(5).*P));
y =fi5;
end



%-----平均瞬时功率---mif--s25--
function y= mif(x)
Fs = 1926;
n = length(x);
t =(0:n-1)/Fs;
dt = mean(diff(t));
analytic_signal = hilbert(x);
amplitude = abs(analytic_signal); %振幅
phase = unwrap(angle(analytic_signal));%相位
instantaneous_frequency = [0; diff(phase)] / (2*pi*dt);
IMFs = emd(x); 
mif = zeros(size(IMFs, 2), 1);
for j = 1:size(IMFs, 2)
    w_j = instantaneous_frequency; % 每个IMF的瞬时频率可能不同，这里简化处理
    mif(j) = sum(w_j .* IMFs(:,j).^2) / sum(IMFs(:,j).^2);
end
MIF = sum(norm(IMFs).* mif / sum(norm(IMFs)));
y= MIF;
end

%-------6-小波分解能量占比----
function y = dwtEnergy(x)
x = x';
Fs = 1926;
wavename = 'db4';   
N = 8; %小波分解水平，正整数
[C, L] = wavedec(x, N, wavename);
[Lo_R, Hi_R] = wfilters(wavename, 'r'); % 获取重构滤波器
A = appcoef(C, L, Lo_R, Hi_R, N);
slen = length(x);
D = zeros(5,slen);
for i = 1:N
    cleanD = detcoef(C, L, N-i+1);
    D(i,:) = [cleanD,zeros(1,slen-length(cleanD))];
end
E = zeros(1,N+1);
E(1) = norm(A,2)^2;
E(2:end) = vecnorm(D,2,2).^2;
E = E/sum(E);
y = E;
end

%--------5--动力学特征----
function y=myentropy(x)
y = entropy(x);
end

function sampEn = sampleEntropy(x, m, r)
    % 输入参数：
    % signal - 输入的时间序列
    % m - 嵌入维度
    % r - 相似度容忍阈值，通常是标准差的倍数
    
    n = length(x);
    A = 0;
    B = 0;
    
    % 计算每一对向量序列
    for i = 1:(n - m)
        for j = 1:(n - m)
            if i ~= j
                if max(abs(x(i:i+m-1) - x(j:j+m-1))) < r
                    B = B + 1;
                    if max(abs(x(i:i+m) - x(j:j+m))) < r
                        A = A + 1;
                    end
                end
            end
        end
    end
    
    % 计算A和B的比值的自然对数，得到SampEn
    sampEn = -log(A / B);
end


%------1--时域特征
function y=wamp(x,thred)
count = 0;
nx = length(x);
for k = 1:nx-1
    x0 = x(k);
    x1 = x(k+1);
    if abs(x0-x1)>thred
        count = count+1;
    end
end
y=count;
end

function y=mav(x)  % =IEMG
y=mean(abs(x));
end

function y=ssc(x,thred)
count = 0;
nx = length(x);
for k = 2:nx-2
    x0 = x(k);
    x1 = x(k-1);
    x2 = x(k+1);
    if(x1-x0)*(x0-x2)>=0||(abs(x1-x0)<thred&&abs(x0-x2)<thred)
    else
        count = count+1;
    end
end
y=count;
end

function y=zerocrossing(x)
count = 0;
nx = length(x);
for k = 1:nx-1
    x0 = x(k);
    x1 = x(k+1);
    if x0*x1<0
        count = count+1;
    end
end
y=count;
end

function y=rsd1(x)
y1 = diff(x);
y = sum(y1.^2);
end

function y=rsd2(x)
y1 = diff(x);
y2 = diff(y1);
y = sum(y2.^2);
end

function y=sis(x) % integral square
y=sum(x.^2);
end

function y=sumRoot(x)
y=real(sum(sqrt(x)));
end

function y=meanExp(x)
y=mean(exp(x));
end

function y=wavelen(x)  %% 肌电复杂程度
y1 = diff(x);
y = sum(abs(y1));
end

function y=iEMG(x)   %% 肌电活动程度
y = sum(abs(x));
end

function y=meandf(x)  %￥差分均值
diffx = diff(x);
y = mean(abs(diffx));
end


function y= PeakFrequency(x)
[~,P] = emgDFT(x);
[~,index] = max(P);
PKF = index;
y = PKF;
end

function y=RMS(x)
    y = sqrt(mean(x.^2));
end

function y=varr(x)
    y = var(x);
end

function y = LogDetector(x)
N = length(x);
y1 = exp(mean(log(x)));
y = abs(y1);
end

function y = vorder(x)
N = length(x);
v=0.1;
y= (sum(abs(x).^v)/N)^(1/v);
end



%------4--模型参数特征--------
function y = TDPSD(x)    %---f1 
%%%%%%%%%%%% 计算avm
avm0 = sqrt(sum(x.^2));
diff1 = diff(x);
diff2 = diff(diff1);
avm2 = sqrt(sum(diff1.^2));
avm4 = sqrt(sum(diff2.^2));
%%%%%%%%%%%%% 归一化
m0 = avm0.^0.1/0.1;
m2 = avm2.^0.1/0.1;
m4 = avm4.^0.1/0.1;
%%%%%%%%%%%%% 计算f1，f2，f3
f1 = log(m0);            %--------其实就是均方根
f2 = log(m0-m2);
f3 = log(m0-m4);
%%%%%%%%%%%% f4:sparseness
f4 = log(m0/(sqrt(m0-m2)*sqrt(m0-m4)));
%%%%%%%%%%%% f5:IF
f5 = m2/sqrt(m0*m4);
%%%%%%%%%%%% f6:WL Ratio
f6 = log(sum(abs(diff1))/sum(abs(diff2)));
y =[f1,f2,f3,f4,f5,f6];
end

function y = ar_coef(x, k)
    n = length(x);
    X = zeros(n-k, k);
    y = x(k+1:n);
    
    % 填充设计矩阵
    for i = 1:k
        X(:, i) = x(k-i+1:n-i);
    end
    
    % 使用最小二乘法求解
    y = X \ y;
end



%--------2--频域特征
function y = MedianFrequency(x)  %MDF
[f,P] = emgDFT(x);
tmp=0;
for i = 1:length(f)
    tmp = tmp+P(i);
    if(tmp>(sum(P)/2))
        break;
    end
end
MDF = f(i);
y = MDF;
end

function y= MeanPowerFrequency(x)   % MPF
[f,P] = emgDFT(x); 
numerator = sum(f .* P');
denominator = sum(P);
MPF = numerator / denominator;
y= MPF;
end

function y= PowerSpectralRatio(x)  %PSR
[f,P] = emgDFT(x);
fLow = 200;fHigh=250;
ix = f >= fLow & f <= fHigh;
PSR = sum(P(ix))/sum(P); 
y= PSR;
end


function y = TotalPower(x)     % TTP
[~,P] = emgDFT(x);
TTP = sum(P);
y = TTP;
end


function [f,P] = emgDFT(x)
Fs = 1000;
N = length(x);
X = fft(x);
P2 = abs(X/N).^2;
P1 = P2(1:fix((N-1)/2));
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:length(P1)-1)/N;
P = P1;
end