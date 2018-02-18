function [ d2 ] = blma_distfun( XI, XJ, d, Data, alpha )
%BLMA_DISTFUN Summary of this function goes here
%   Detailed explanation goes here

nD = length(Data);
nU = size(Data{1},1);

if nD > 1
    weight = 0.5/(nD-1)*ones(nD, 1);
    weight(d) = 0.5;
else
    weight = 1;
end

i = XI(1);

nUser = size(XJ,1);

pUsers = XJ(:)';

s = spalloc(nU,nD,nUser*nD);

tic
fprintf('Start computer distance over %d users w.r.t. user %d\n', nUser, i);
for d=1:nD
    data_d = Data{d};
    data_i = data_d(i, :);
    parfor j = pUsers;
        data_j = data_d(j, :);
        coitem = find(data_i & data_j);
        nitem = length(coitem);
        if nitem > 0
            corating_i = data_i(coitem);
            corating_j = data_j(coitem);
            cos = (corating_i*corating_j')/sqrt((corating_i*corating_i')*(corating_j*corating_j'));
            s(j,d) = nitem/(nitem + alpha) * cos;
        end
    end
    
end

d2 = 1 - s(pUsers,:)*weight;

fprintf('Finish computer distance in %g secs\n', toc);

end