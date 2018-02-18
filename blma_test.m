K = 30;
is_commavg = false;
nD = 4;

D.memship = cell(nD,1);

domain='book';
percent=0.75;

Book_Matrix(isnan(Book_Matrix))=0;
Book_Matrix=sparse(Book_Matrix);
Music_Matrix(isnan(Music_Matrix))=0;
Music_Matrix=sparse(Music_Matrix);
DVD_Matrix(isnan(DVD_Matrix))=0;
DVD_Matrix=sparse(DVD_Matrix);
Video_Matrix(isnan(Video_Matrix))=0;
Video_Matrix = sparse(Video_Matrix);

nU = size (Book_Matrix,1);
X = cell(4,1);

SN =Book_Matrix;
if(strcmp(domain,'book'))
    switch percent
        case 0.75
            testIdx=missingInds_book_75;
            trainingInd=trainingInds_book_75;
        case 0.5
            testIdx=missingInds_book_50;
            trainingInd=trainingInds_book_50;
        case 0.2
            testIdx=missingInds_book_20;
            trainingInd=trainingInds_book_20;
        case 0.05
            testIdx=missingInds_book_5;
            trainingInd=trainingInds_book_5;
        case 0
            testIdx=false(size(SN));
            testIdx(ColdStartUsers,:) = SN(ColdStartUsers,:)~=0;
    end
    
    trueVals = nonzeros(SN(testIdx));
    SN(testIdx)=0;
    X{1}=SN;
    D.memship{1} = memship{1};
else
    X{2}=SN;
    D.memship{2} = memship{1};
end

SN=Music_Matrix;
if(strcmp(domain,'music'))
    switch percent
        case 0.75
            testIdx=missingInds_music_75;
            trainingInd=trainingInds_music_75;
        case 0.5
            testIdx=missingInds_music_50;
            trainingInd=trainingInds_music_50;
        case 0.2
            testIdx=missingInds_music_20;
            trainingInd=trainingInds_music_20;
        case 0.05
            testIdx=missingInds_music_5;
            trainingInd=trainingInds_music_5;
        case 0
            testIdx=false(size(SN));
            testIdx(ColdStartUsers,:) = SN(ColdStartUsers,:)~=0;
    end
    trueVals = SN(testIdx);
    SN(testIdx)=0;
    X{1}=SN;
    D.memship{1} = memship{2};
else
    X{2}=SN;
    D.memship{2} = memship{2};
end

[i,j] = find(testIdx);
testIdx_p = [i,j];

X{3}=DVD_Matrix;
D.memship{3} = memship{3};

X{4}=Video_Matrix;
D.memship{4} = memship{4};

D.data = X;
D.mu = cellfun(@(x) mean(nonzeros(x)), D.data);
D.memship = memship;
nC = cellfun(@max, D.memship);

%% Random Initialization
G.effects = 1/nD*rand(K,nU);
G.invSigma = eye(K);

A.effects = 1/nD*rand(K,nD);

O.effects = cell(1,nD);
for i=1:nD
    O.effects{i} = 1/nD*rand(K,nC(i));
end
O.invSigma = cell(1,nD);
O.invSigma(:) = {eye(K)};

S.effects = cell(1,nD);
S.effects(:) = {1/nD*rand(K,nU);};
S.invSigma = cell(1,nD);
S.invSigma(:) = {eye(K)};

V.invSigma = cell(1,nD);
V.invSigma(:) = {eye(K)};
V.effects = cell(1,nD);
for i=1:nD
    V.effects{i} = D.mu(i)/K*rand(K,size(X{i},2));
end

sigma2 = 1;
epsilon = 1e-3;

%% Burn-in iterations
nItr = 5;
for i=1:5
    [ V,A,O,S,G,sigma2 ] = blma(D,K,V,A,O,S,G,sigma2,epsilon,nItr,2);
    [ predvals ] = blma_predict( testIdx_p,...
                V.effects{1},A.effects(:,1),O.effects{1},S.effects{1},G.effects,...
                D.mu(1),D.memship{1},is_commavg);
    predvals(predvals>5) = 5;
    predvals(predvals<1) = 1;
    mae = sum(abs( trueVals - predvals))/numel(trueVals);
    fprintf('Burn-in: %d iterations finished. Probe MAE: %g\n', nItr*i, mae);
end

nSamp = 20;
%% Draw 10 samples, lag = 10
predvals_popmean = 0;
predvals_commean = 0;
lag = 1;
for i=1:nSamp
    fprintf('Drawing sample %d\n', i);
    [ V,A,O,S,G,sigma2 ] = blma(D,K,V,A,O,S,G,sigma2,epsilon,lag,2);
    [ result ] = blma_predict( testIdx_p,...
                V.effects{1},A.effects(:,1),O.effects{1},S.effects{1},G.effects,...
                D.mu(1),D.memship{1},false);
     predvals_popmean = predvals_popmean + result;
     [ result ] = blma_predict( testIdx_p,...
                V.effects{1},A.effects(:,1),O.effects{1},S.effects{1},G.effects,...
                D.mu(1),D.memship{1},true);
     predvals_commean = predvals_commean + result;
end

predvals_popmean = predvals_popmean/nSamp;
predvals_popmean(predvals_popmean>5) = 5;
predvals_popmean(predvals_popmean<1) = 1;
mae_popmean = sum(abs( trueVals - predvals_popmean))/numel(trueVals);

predvals_commean = predvals_commean/nSamp;
predvals_commean(predvals_commean>5) = 5;
predvals_commean(predvals_commean<1) = 1;
mae_commean = sum(abs( trueVals - predvals_commean))/numel(trueVals);

fprintf('MAE-PopMean: %g, MAE-ComMean: %g\n', mae_popmean, mae_commean);