function y=max_getac(data)
win = 8000;
iend = size(data,2);
mend = size(data,1);
sqww = [];
for i = 1
    sqw = [];
    for m = 1:mend-8000
        td = data(m:m+8000,i);
        sq = sum(abs(td));
        sqw = [sqw;sq];
    end
    sqww = [sqww sqw];
end
Fs = 1926; 
Wn = [1]/(Fs/2);
[b, a] = butter(5, Wn, 'low'); 
flag= filtfilt(b, a, sqww);
%---------
flag = flag(10000:end);
[~,loc1]=findpeaks(flag);
loc2 = loc1(1);
for i = 2:length(loc1)
    if (loc1(i)-loc2(end))>15000
        loc2 = [loc2,loc1(i)];
    end
end
%----------check
% plot(flag)
% hold on 
% plot(loc2,flag(loc2),'r*')
x1 = loc2;
x2 = loc2+8000;
y=[];
for i = 1:length(x1)
ty = data(x1(i):x2(i),:);
y = [y;ty];
end
p=0;
end