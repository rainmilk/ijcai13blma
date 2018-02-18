function [ result ] = blma_predict( testIdx_d,v_d,a_d,o_d,s_d,g,mu_d,memship_d,is_commavg)
%PREDICT Summary of this function goes here
%   Detailed explanation goes here
if ~exist('is_commavg','var')
    is_commavg = false;
end

%% Avg non-coldstart user factors
if is_commavg
    c_d = zeros(size(o_d));
    nC = size(o_d, 2);
    nNonCS = 0;
    for c = 1:nC
        user = find(memship_d == c);
        nu = length(user);
        for ui=1:nu
            i = user(ui);
            if any(s_d(:,i))
                c_d(:,c) = c_d(:,c) + s_d(:,i);
                nNonCS = nNonCS + 1;
            end
        end
        if nNonCS > 0
            c_d(:,c) = c_d(:,c)/nNonCS;
            for ui=1:nu
                i = user(ui);
                if ~any(s_d(:,i))
                    s_d(:,i) = c_d(:,c);
                end
            end
        end
    end
end

%% Generate U factors
nU = size(s_d, 2);
u_d = zeros(size(s_d));
for i = 1:nU
    ci = memship_d(i);
    u_d(:,i) = a_d + o_d(:,ci)+ s_d(:,i) + g(:,i);
end

%% Prediction
nTestData = size(testIdx_d, 1);
result = zeros(nTestData, 1);
for t=1:nTestData
    pair = testIdx_d(t,:);
    i = pair(1);
    j = pair(2);
    result(t) = mu_d + v_d(:,j)'*u_d(:,i);
end
end

