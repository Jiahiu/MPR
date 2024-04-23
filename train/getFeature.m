function [feat,dataclass] = getFeature(data)
winlen = 200;               
inc = 150;
[nx,ny] = size(data{1});
num = floor((nx-winlen)/inc);
feat2 = [];
dataclass = [];
for k = 1:8
    thisData = data{k};
    feat_class = [];
    for i = 1:num

        feats_achan = [];
        for j = 1:ny
            feats = [];
            x = thisData(1+(i-1)*inc:winlen+(i-1)*inc,j);
            tdpsd=FeatsFun(x,6);
            rms = FeatsFun(x,1);
            mav=FeatsFun(x,22);
            mpf=FeatsFun(x,7);
            dasdv=FeatsFun(x,12);
            mdf=FeatsFun(x,11); 
            % ar4 = FeatsFun(x,25);
            % sampEn = FeatsFun(x,26);
            feats = [rms,mav,dasdv,mpf,mdf,tdpsd(1)];
            % tt = FeatsFun(x,29);
            % feats = [tt];
            % dwtEn = FeatsFun(x,27);
            % feats = dwtEn(4:9);
            % feats = [rms,mav,mpf,mdf];
            % feats = feats(2);
            feats_achan = [feats_achan,feats];
        end
        feat_class = [feat_class;feats_achan];
    end
    feat2 = [feat2;feat_class];
    dataclass = [dataclass;ones(num,1)*k];
end
%--- 归一化
feat = zscore(feat2);
end