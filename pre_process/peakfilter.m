function y=peakfilter(data)
iend = size(data,2);
rd = [];
for i = 1:iend
    td = data(:,i);
    lamda = 1e-4;
    ix = find(abs(td)>lamda);
    td(ix) = randn(1,length(ix))*1e-5;
    rd = [rd td];
end
y=rd;
end