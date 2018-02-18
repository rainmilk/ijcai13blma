Book_Matrix(isnan(Book_Matrix))=0;
Book_Matrix=sparse(Book_Matrix);
Music_Matrix(isnan(Music_Matrix))=0;
Music_Matrix=sparse(Music_Matrix);
DVD_Matrix(isnan(DVD_Matrix))=0;
DVD_Matrix=sparse(DVD_Matrix);
Video_Matrix(isnan(Video_Matrix))=0;
Video_Matrix = sparse(Video_Matrix);
Data = {Book_Matrix, Music_Matrix, DVD_Matrix, Video_Matrix};

nD = length(Data);

X = 1:size(Data{1},1);
X = X';
alpha = 50;
T = cell(1, nD);
for d = 1:nD
    distfun = @(XI, XJ) blma_distfun( XI, XJ, d, Data, alpha );
    Y = pdist(X,distfun);
    Z = linkage(Y,'complete'); 
    T{d} = cluster(Z,'maxclust',50); 
end