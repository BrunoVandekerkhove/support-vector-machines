% Dataset
load digits; clear size
[N, dim] = size(X);
testset = Xtest2;
Ntest = size(testset,1);
minx = min(min(X)); 
maxx = max(max(X));

% Add noise
noisefactor = 1.0;
noise = noisefactor * maxx; % sd for Gaussian noise
Xn = X;
for i = 1:N
  randn('state', i);
  Xn(i,:) = X(i,:) + noise * randn(1, dim);
end
Xnt = testset; 
for i = 1:size(testset,1)
  randn('state', N+i);
  Xnt(i,:) = testset(i,:) + noise * randn(1,dim);
end

% Select training set
Xtr = X(1:1:end,:);

npc_comb = [1,2,4,8,16,32,64,128,198];
sigma_comb = [0.005, 0.01, 7.^(-2:0.2:2)];

allerrors = zeros(length(npc_comb),length(sigma_comb));

idx = 0;

for npcs = npc_comb

    idx = idx + 1;
    jdx = 0;
    
for sigmafactor = sigma_comb
    
    jdx = jdx + 1;

% Hyperparameters
sig2 =dim*mean(var(Xtr)); % rule of thumb
sig2 = sig2*sigmafactor;

% KPCA
[lam,U] = kpca(Xtr, 'RBF_kernel', sig2, [], 'eig', 240); 
Ud=U(:,(1:npcs));
[lam, ids] = sort(-lam); lam = -lam; U = U(:,ids);
%[lam,U] = kpca(Xtr,'RBF_kernel',sig2);
%[lam, perm] = sort(-lam); lam = -lam; U = U(:,perm);
errors = [];
to_reconstruct = Xtr; % testset
%projections = kernel_matrix(Xtr, 'RBF_kernel', sig2, eval(to_reconstruct))' * U;
Xr = preimage_rbf(Xtr,sig2,Ud,Xn,'denoise'); %Xtn
%Xr = preimage_rbf(Xtr, sig2, U, projections(:,1:npcs), 'r'); % reconstruction
error = norm(Xr - to_reconstruct, 2);
allerrors(idx,jdx) = error %#ok

% Visualisation
VISUALIZE = 0;
if VISUALIZE
    digs = [0:9]; ndig = length(digs); % choose the digits for test
    m=2; % choose the mth data for each digit 
    Xdt=zeros(ndig, dim);
    figure; 
    colormap('gray'); 
    title('Denosing using kPCA'); tic
    npcs = [2.^(0:7) 190];
    lpcs = length(npcs);
    for k=1:lpcs;
     nb_pcs=npcs(k); 
     disp(['nb_pcs = ', num2str(nb_pcs)]); 
     Ud=U(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
     for i=1:ndig
       dig=digs(i);
       fprintf('digit %d : ', dig)
       xt=Xnt(i,:);
       if k==1 
         % plot the original clean digits
         %
         subplot(2+lpcs, ndig, i);
         pcolor(1:15,16:-1:1,reshape(Xtest2(i,:), 15, 16)'); shading interp; 
         set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
         if i==1, ylabel('original'), end 
         % plot the noisy digits 
         %
         subplot(2+lpcs, ndig, i+ndig); 
         pcolor(1:15,16:-1:1,reshape(xt, 15, 16)'); shading interp; 
         set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);        
         if i==1, ylabel('noisy'), end
         drawnow
       end    
       Xdt(i,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
       subplot(2+lpcs, ndig, i+(2+k-1)*ndig);
       pcolor(1:15,16:-1:1,reshape(Xdt(i,:), 15, 16)'); shading interp; 
       set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);           
       if i==1, ylabel(['n=',num2str(nb_pcs)]); end
       drawnow    
     end % for i
    end % for k
end

end

end
%%

usps_errors = zeros(10,21,3);

j = 0;
truej = 0;
for i = 1:10
    for aaa = 1:21
        j = j + 1;
        truej = truej + 1;
        if truej > 21
            truej = 1;
        end
        usps_errors(i,truej,:) = allerrors(i,j,:);
    end
end

%%
figure
for i = 1:size(Xtr,1)/2
    subplot(11,9,i)
    pcolor(1:15,16:-1:1,reshape(Xtr(i,:), 15, 16)');
    shading interp; 
    colormap gray;
    drawnow
end
%% Contour plot

vals = allerrors(:,:,1);
vals = (vals - (min(min(vals))))/(max(max(vals)) - min(min(vals)));
vals = 1-sqrt(vals);

[X,Y] = meshgrid(sigma_comb, npc_comb);
[Xq,Yq] = meshgrid(min(sigma_comb):0.1:max(sigma_comb), 1:198);
Vq = interp2(X, Y, vals, Xq, Yq);

figure
contourf(Xq,Yq,Vq); % Interpolated version
% contourf(sigma_comb, npc_comb, vals)
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')