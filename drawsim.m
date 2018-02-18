% [ Vm,Am,Om,Sm,Gm,sigma2m ] = blma_mean(D,K,V,A,O,S,G,sigma2,epsilon);
% Draw domain effects
[K,nD] = size(Am.effects);
for i=1:nD
    subplot(nD, 2, 2*i-1);
    stem(1:K, Am.effects(:,i));
end

% Draw similarities
subplot(nD, 2, [2 4]);
mat = zeros(nD);
for i=1:3
    for j=i+1:nD
        mat(i,j) = Am.effects(:,i)'*Am.effects(:,j)/sqrt((Am.effects(:,i)'*Am.effects(:,i))*(Am.effects(:,j)'*Am.effects(:,j)));
    end
end
mat = mat + mat';
mat = mat + eye(4);

imagesc(mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(mat(:),'%1.3f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:size(mat,2),1:size(mat,1));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings,...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:4,'XTickLabel',{'Book','Music','DVD','VHS'},...                         %# Change the axes tick marks
        'YTick',1:4,'YTickLabel',{'Book','Music','DVD','VHS'});
    
% Draw vairance for community effects
subplot(nD, 2, 6);
stem(1:K, diag(inv(Om.invSigma{1}),0));
subplot(nD, 2, 8);
stem(1:K, diag(inv(Om.invSigma{2}),0));