function [ V,A,O,S,G,sigma2 ] = blma_mean(D,K,V,A,O,S,G,sigma2,epsilon)
%BLMA Summary of this function goes here
% D.data denotes the cells containing ratings for each domain
% D.memship denotes the cells containing memberships for each domain

if ~exist('epsilon','var')
    epsilon = 1e-2;
end

% the number of domains
nD = length(D.data);
% the number of users
nU = size(D.data{1},1);
% the number of observations
nData = sum(cellfun(@nnz, D.data));

V = chkhyperparam(V,nD,K);
O = chkhyperparam(O,nD,K);
S = chkhyperparam(S,nD,K);
G = chkhyperparam(G,1,K);

%% Set Default Values
if ~isfield(D,'mu')
    mu = cellfun(@(x) mean(nonzeros(x)), D.data);
else
    mu = D.mu;
end

data = D.data;
memship = D.memship;

a_E = A.effects;
o_E = O.effects;
s_E = S.effects;
g_E = G.effects;
v_E = V.effects;
a_mE = a_E;
o_mE = o_E;
s_mE = s_E;
g_mE = g_E;
v_mE = v_E;
invSigma_o = O.invSigma;
invSigma_s = S.invSigma;
invSigma_g = G.invSigma;
invSigma_v = V.invSigma;
invSigma_mo = invSigma_o;
invSigma_ms = invSigma_s;
invSigma_mv = invSigma_v;


U = cell(nD, 1);

parfor d=1:nD
    data_d = data{d};
    memship_d = memship{d};
    o_d = o_E{d};
    s_d = s_E{d};
    v_d = v_E{d};
    mu_d = mu(d);
    sumSigma = 0;
    sumMu = 0;
    [user,item] = find(data_d);
    n_d_Data = length(item);
    for r=1:n_d_Data
        i = user(r);
        j = item(r);
        v_dj = v_d(:,j);
        sumSigma = sumSigma + v_dj*v_dj';
        ci = memship_d(i);
        sumMu = sumMu + v_dj*(data_d(i,j) - mu_d - v_dj'*(o_d(:,ci)+s_d(:,i)+g_E(:,i)));
    end
    sigma_d_hat = inv(sumSigma/sigma2);
    a_mE(:,d) = sigma_d_hat*(sumMu/sigma2);
end


parfor d=1:nD
    data_d = data{d};
    memship_d = memship{d};
    a_d = a_E(:,d);
    o_d = o_E{d};
    s_d = s_E{d};
    v_d = v_E{d};
    mu_d = mu(d);
    invSigma_d = invSigma_o{d};
    ndC = size(o_d,2);
    for c = 1:ndC
        ncij = 0;
        sumSigma = 0;
        sumMu = 0;
        user = find(memship_d==c);
        nu = length(user);
        for ui = 1:nu
            i = user(ui);
            s_di_g_i = s_d(:,i) + g_E(:,i);
            item = find(data_d(i,:));
            ni = length(item);
            ncij = ncij + ni;
            for ii = 1:ni
                j = item(ii);
                v_dj = v_d(:,j);
                sumSigma = sumSigma + v_dj*v_dj';
                sumMu = sumMu + v_dj*(data_d(i,j) - mu_d - v_dj'*(a_d + s_di_g_i));
            end
        end
        if ncij > 0
            sigma_d_hat = inv(sumSigma/sigma2 + invSigma_d);
            o_d(:,c) = sigma_d_hat*(sumMu/sigma2);
        else
            o_d(:,c) = zeros(K, 1);
        end
    end
    o_mE{d} = o_d;
end

sumSigma = cell(nU,1);
sumSigma(:) = {0};
sumMu = cell(nU,1);
sumMu(:) = {0};
for d=1:nD
    data_d = data{d};
    memship_d = memship{d};
    a_d = a_E(:,d);
    o_d = o_E{d};
    s_d = s_E{d};
    v_d = v_E{d};
    mu_d = mu(d);
    parfor i = 1:nU
        data_di = data_d(i,:);
        item = find(data_di);
        ni = length(item);
        for ii = 1:ni
            j = item(ii);
            v_dj = v_d(:,j);
            sumSigma{i} = sumSigma{i} + v_dj*v_dj';
            ci = memship_d(i);
            sumMu{i} = sumMu{i} + v_dj*(data_di(j) - mu_d - v_dj'*(a_d + o_d(:,ci)+s_d(:,i)));
        end
    end
end
parfor i = 1:nU
    sigma_hat = inv(sumSigma{i}/sigma2 + invSigma_g);
    g_mE(:,i) = sigma_hat*(sumMu{i}/sigma2);
end

for d=1:nD
    data_d = data{d};
    memship_d = memship{d};
    a_d = a_E(:,d);
    o_d = o_E{d};
    s_d = s_E{d};
    v_d = v_E{d};
    invSigma_d = invSigma_s{d};
    mu_d = mu(d);
    parfor i = 1:nU
        sumSigma = 0;
        sumMu = 0;
        data_di = data_d(i,:);
        item = find(data_di);
        ni = length(item);
        for ii = 1:ni
            j = item(ii);
            v_dj = v_d(:,j);
            sumSigma = sumSigma + v_dj*v_dj';
            ci = memship_d(i);
            sumMu = sumMu + v_dj*(data_di(j) - mu_d - v_dj'*(a_d + o_d(:,ci)+g_E(:,i)));
        end
        if ni > 0
            sigma_d_hat = inv(sumSigma/sigma2 + invSigma_d);
            s_d(:,i) = sigma_d_hat*(sumMu/sigma2);
        else
            s_d(:,i) = zeros(K,1);
        end
    end
    s_mE{d} = s_d;
end

% Sample inverse sigma_g
sumSigma = 0;
parfor i = 1:nU
    g_i = g_E(:,i);
    sumSigma = sumSigma + g_i*g_i';
end
nu = G.nu + nU;
phi = G.phi + sumSigma;
invSigma_mg = inv(phi/(nu-K-1));

for d=1:nD
    memship_d = memship{d};
    a_d = a_E(:,d);
    o_d = o_E{d};
    s_d = s_E{d};
    ndC = size(o_d,2);
    
    % Sample inverse sigma_od
    sumSigma = 0;
    parfor c = 1:ndC
        o_dc = o_d(:,c);
        sumSigma = sumSigma + o_dc*o_dc';
    end
    nu = O.nu(d) + ndC;
    phi = O.phi{d} + sumSigma;
    invSigma_mo{d} = inv(phi/(nu-K-1));
    
    u_d = zeros(K,nU);
    sumSigma = 0;
    ndU = 0;
    parfor i = 1:nU
        s_di = s_d(:,i);
        sumSigma = sumSigma + s_di*s_di';
        
        % Generate U factors
        ci = memship_d(i);
        u_di = a_d + o_d(:,ci)+ s_di +g_E(:,i);
        u_d(:,i) = u_di;
    end
    % Sample inverse sigma_sd
    nu = S.nu(d) + ndU;
    phi = S.phi{d} + sumSigma;
    invSigma_ms{d} = inv(phi/(nu-K-1));
    
    U{d} = u_d;
end


for d=1:nD
    data_d = data{d};
    mu_d = mu(d);
    v_d = v_E{d};
    u_d = U{d};
    invSigma_d = invSigma_v{d};
    ndI = size(v_d, 2);
    parfor j = 1:ndI
        sumSigma = 0;
        sumMu = 0;
        data_dj = data_d(:,j);
        user = find(data_dj);
        nu = length(user);
        for ui = 1:nu
            i = user(ui);
            u_di = u_d(:,i);
            sumSigma = sumSigma + u_di*u_di';
            sumMu = sumMu + u_di*(data_dj(i) - mu_d);
        end
        if nu > 0
            sigma_d_hat = inv(sumSigma/sigma2 + invSigma_d);
            v_d(:,j) = sigma_d_hat*(sumMu/sigma2);
        else
            v_d(:,j) = zeros(K,1);
        end
    end
    v_mE{d} = v_d;
end


sumerror2 = 0;
for d=1:nD
    nI = 0;
    sumSigma = zeros(K);
    data_d = data{d};
    mu_d = mu(d);
    v_d = v_E{d};
    u_d = U{d};
    ndI = size(v_d, 2);
    parfor j = 1:ndI
        data_dj = data_d(:,j);
        user = find(data_dj);
        nu = length(user);
        if nu > 0
            nI = nI + 1;
            v_dj = v_d(:,j);
            sumSigma = sumSigma + v_dj*v_dj';
            
            % Sum error squre
            for ui = 1:nu
                i = user(ui);
                u_di = u_d(:,i);
                error = data_dj(i) - mu_d - v_dj'*u_di;
                sumerror2 = sumerror2 + error*error;
            end
        end
    end
    % Sample inverse sigma_d_v
    nu = V.nu(d) + nI;
    phi = V.phi{d} + sumSigma;
    invSigma_mv{d} = inv(phi/(nu-K-1));
end

% Inverse gamma
sigma2 = (epsilon + sumerror2/2)/(epsilon + nData/2 - 1);

%% Set outputs
A.effects = a_mE;
O.effects = o_mE;
S.effects = s_mE;
G.effects = g_mE;
V.effects = v_mE;
O.invSigma = invSigma_mo;
S.invSigma = invSigma_ms;
G.invSigma = invSigma_mg;
V.invSigma = invSigma_mv;
end

function X = chkhyperparam(X,n,K)
if ~isfield(X,'phi')
    if n>1
        X.phi = cell(n,1);
        X.phi(:) = {eye(K)};
    else
        X.phi = eye(K);
    end
end

if ~isfield('X','nu')
    X.nu = K*ones(n,1);
end
end

