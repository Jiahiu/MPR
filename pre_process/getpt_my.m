function y = getpt_my(data,k)
figure(1);
set(gcf,'color',[1 1 1]);

% disp(['motion now is',' ', num2str(k)])
for ch=1:12
    plot(data(:,ch));
    num = num2str(ch);
    titleName = ['ch',num,' motion ',num2str(k)];
    title(titleName)
    
    Next=input('Next Channels?','s');
    if Next~='y'
        break;
    end
    if ch==12
        ch=1;
    end
end

plot(data(:,ch));
[x1,y1]=getpts;

x1 = fix(x1);
x2 = x1+8e3-1;
if length(x1)<4
    y = [];
else 
    y = [x1 x2];
end
end