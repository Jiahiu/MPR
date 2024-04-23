function y=bdfilter(data)
    Fs = 1926; 
    Wn = [10 450]/(Fs/2);
    [b, a] = butter(5, Wn, 'bandpass'); 
    y= filtfilt(b, a, data);
end