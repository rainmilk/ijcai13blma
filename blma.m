function [ V,A,O,S,G,sigma2 ] = blma(D,K,V,A,O,S,G,sigma2,epsilon,nIter,subIter)
%BLMA Summary of this function goes here
% D.data denotes the cells containing ratings for each domain
% D.memship denotes the cells containing memberships for each domain

if ~exist('epsilon','var')
    epsilon = 1e-2;
end

if ~exist('subIter','var')
    subIter = 2;
end

if ~exist('nIter','var')
    nIter = 100;
end

if length(subIter) == 1
    subIter = [subIter subIter];
end

% the number of domains
nD = length(D.data);
% the number of users
nU = size(D.data{1},1);
% the number of observations
nData = sum(cellfun(@nnz, D.data));

%% Set Default Values
if ~isfield(D,'mu')
    mu = cellfun(@(x) mean(nonzeros(x)), D.data);
else
    mu = D.mu;
end

V = chkhyperparam(V,nD,K);
O = chkhyperparam(O,nD,K);
S = chkhyperparam(S,nD,K);
G = chkhyperparam(G,1,K);

data = D.data;
memship = D.memship;

a_E = A.effects;
o_E = O.effects;
s_E = S.effects;
g_E = G.effects;
v_E = V.effects;
invSigma_o = O.invSigma;
invSigma_s = S.invSigma;
invSigma_g = G.invSigma;
invSigma_v = V.invSigma;

U = cell(nD, 1);

%% Start Iteration
it = 1;
tLast = 0;
ticID = tic;
while it <= nIter
    fprintf('Start running iteration %d:\n', it);
    
    sumerror2_p = 0;
    for uit=1:subIter(1)
        fprintf('Fitting LMM w.r.t. U, Sub-iteration %d:\n', uit);
        %% For d=1:nD, sample the domain effects A.effects in parallel
        fprintf('Start sampling domain effects...\n');
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
            ad_hat = sigma_d_hat*(sumMu/sigma2);
            a_E(:,d) = mvnrnd(ad_hat',sigma_d_hat);
        end
        
        %         fprintf('Debug: Domain effects varainces: %g\n', ...
        %         debugVariance(nD, nData, data, mu, memship, a_E, o_E, s_E, g_E, v_E));
        
        tElapsed = toc(ticID);
        fprintf('Finished sampling domain effects in %g secs\n', tElapsed - tLast);
        tLast = tElapsed;
        
        
        %% For c=1:ndC, sample the community effects o_dc in parallel for each domain d
        fprintf('Start sampling community effects...\n');
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
                    od_hat = sigma_d_hat*(sumMu/sigma2);
                    o_d(:,c) = mvnrnd(od_hat',sigma_d_hat);
                else
                    o_d(:,c) = zeros(K, 1);
                end
            end
            o_E{d} = o_d;
        end
        
        %         fprintf('Debug: Community effects varainces: %g\n', ...
        %         debugVariance(nD, nData, data, mu, memship, a_E, o_E, s_E, g_E, v_E));
        
        tElapsed = toc(ticID);
        fprintf('Finished sampling community effects in %g secs\n', tElapsed - tLast);
        tLast = tElapsed;
        
        %% For g=1:nU, sample the cross-domain user effects g_i in parallel
        fprintf('Start sampling global user effects...\n');
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
            g_hat = sigma_hat*(sumMu{i}/sigma2);
            g_E(:,i) = mvnrnd(g_hat',sigma_hat);
        end
        
        %         fprintf('Debug: Global user effects varainces: %g\n', ...
        %         debugVariance(nD, nData, data, mu, memship, a_E, o_E, s_E, g_E, v_E));
        
        tElapsed = toc(ticID);
        fprintf('Finished sampling global user effects in %g secs\n', tElapsed - tLast);
        tLast = tElapsed;
        
        %% For s=1:nU, sample the domian-specific user effects s_di in parallel for each domain d
        fprintf('Start sampling domian-specific user effects...\n');
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
                    sd_hat = sigma_d_hat*(sumMu/sigma2);
                    s_d(:,i) = mvnrnd(sd_hat',sigma_d_hat);
                else
                    s_d(:,i) = zeros(K,1);
                end
            end
            s_E{d} = s_d;
        end
        
        %         fprintf('Debug: Domain-specific user effects varainces: %g\n', ...
        %         debugVariance(nD, nData, data, mu, memship, a_E, o_E, s_E, g_E, v_E));
        
        tElapsed = toc(ticID);
        fprintf('Finished sampling domian-specific user effects in %g secs\n', tElapsed - tLast);
        tLast = tElapsed;
        
        %% Sample variance parameters, {sigma_o,sigma_s,sigma_g,sigma2}
        fprintf('Start sampling variance parameters...\n');
        % Sample inverse sigma_g
        sumSigma = 0;
        parfor i = 1:nU
            g_i = g_E(:,i);
            sumSigma = sumSigma + g_i*g_i';
        end
        nu = G.nu + nU;
        phi = G.phi + sumSigma;
        invSigma_g = inv(iwishrnd(phi, nu));
        
        sumerror2 = 0;
        for d=1:nD
            data_d = data{d};
            memship_d = memship{d};
            a_d = a_E(:,d);
            o_d = o_E{d};
            s_d = s_E{d};
            v_d = v_E{d};
            ndC = size(o_d,2);
            mu_d = mu(d);
            
            % Sample inverse sigma_od
            sumSigma = 0;
            parfor c = 1:ndC
                o_dc = o_d(:,c);
                sumSigma = sumSigma + o_dc*o_dc';
            end
            nu = O.nu(d) + ndC;
            phi = O.phi{d} + sumSigma;
            invSigma_o{d} = inv(iwishrnd(phi, nu));
            
            u_d = zeros(K,nU);
            sumSigma = 0;
            ndU = 0;
            parfor i = 1:nU                
                % Generate U factors
                s_di = s_d(:,i);
                ci = memship_d(i);
                u_di = a_d + o_d(:,ci) + s_di + g_E(:,i);
                u_d(:,i) = u_di;
                
                % Sum error squre
                data_di = data_d(i,:);
                item = find(data_di);
                ni = length(item);
                if ni > 0
                    ndU = ndU + 1;
                    sumSigma = sumSigma + s_di*s_di';
                    
                    for ii = 1:ni
                        j = item(ii);
                        v_dj = v_d(:,j);
                        error = data_di(j) - mu_d - v_dj'*u_di;
                        sumerror2 = sumerror2 + error*error;
                    end
                end
            end
            % Sample inverse sigma_sd
            nu = S.nu(d) + ndU;
            phi = S.phi{d} + sumSigma;
            invSigma_s{d} = inv(iwishrnd(phi, nu));
            
            U{d} = u_d;
        end
        
        % Inverse gamma
        sigma2 = 1/gamrnd(epsilon + nData/2, 1/(epsilon + sumerror2/2));
        
        tElapsed = toc(ticID);
        fprintf('Finished sampling variance parameters in %g secs\n', tElapsed - tLast);
        tLast = tElapsed;
        
        tol = abs((sumerror2_p-sumerror2)/sumerror2);
        fprintf('Error square fitting users: %g, Variance: %g, Tol:%d\n',sumerror2, sigma2, tol);
        sumerror2_p = sumerror2;
    end
    
    
    sumerror2_p = 0;
    for uit=1:subIter(2)
        fprintf('Fitting LMM w.r.t. V, Sub-iteration: %d:\n', uit);
        %% For j=1:ndC, sample the item effects v_dj in parallel for each domain d
        fprintf('Start sampling item effects...\n');
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
                    vd_hat = sigma_d_hat*(sumMu/sigma2);
                    v_d(:,j) = mvnrnd(vd_hat',sigma_d_hat);
                else
                    v_d(:,j) = zeros(K,1);
                end
            end
            v_E{d} = v_d;
        end
        
        tElapsed = toc(ticID);
        fprintf('Finished item effects in %g secs\n', tElapsed - tLast);
        tLast = tElapsed;
        
        %% Sample variance parameters, {sigma_v,sigma2}
        fprintf('Start variance parameters...\n');
        
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
            invSigma_v{d} = inv(iwishrnd(phi, nu));
        end
        
        % Inverse gamma
        sigma2 = 1/gamrnd(epsilon + nData/2, 1/(epsilon + sumerror2/2));
        
        tElapsed = toc(ticID);
        fprintf('Finished variance parameters in %g secs\n', tElapsed - tLast);
        tLast = tElapsed;
        
        tol = abs((sumerror2_p-sumerror2)/sumerror2);
        fprintf('Error square fitting items: %g, Variance: %g, Tol:%d\n',sumerror2, sigma2, tol);
        sumerror2_p = sumerror2;
    end
    
    
    %% Iteration Finished
    avgit = tElapsed/it;
    fprintf('Finished iterations (%d/%d), Avg. %g secs per iteration, Est. to finish in %g secs\n',...
        it, nIter, avgit, avgit*(nIter-it));
    
    it = it + 1;
end


%% Set outputs
A.effects = a_E;
O.effects = o_E;
S.effects = s_E;
G.effects = g_E;
V.effects = v_E;
O.invSigma = invSigma_o;
S.invSigma = invSigma_s;
G.invSigma = invSigma_g;
V.invSigma = invSigma_v;
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

function sigma2 = debugVariance(nD, nData, data, mu, memship, a_E, o_E, s_E, g_E, v_E)
sumerror2 = 0;
nU = size(s_E,1);
for d=1:nD
    data_d = data{d};
    memship_d = memship{d};
    a_d = a_E(:,d);
    o_d = o_E{d};
    s_d = s_E{d};
    v_d = v_E{d};
    mu_d = mu(d);
    
    parfor i = 1:nU
        s_di = s_d(:,i);
        
        % Generate U factors
        ci = memship_d(i);
        u_di = a_d + o_d(:,ci)+ s_di +g_E(:,i);
        
        % Sum error squre
        data_di = data_d(i,:);
        item = find(data_di);
        ni = length(item);
        for ii = 1:ni
            j = item(ii);
            v_dj = v_d(:,j);
            error = data_di(j) - mu_d - v_dj'*u_di;
            sumerror2 = sumerror2 + error*error;
        end
    end
end

% Mean sigma2
sigma2 = sumerror2/(nData-1);
end
