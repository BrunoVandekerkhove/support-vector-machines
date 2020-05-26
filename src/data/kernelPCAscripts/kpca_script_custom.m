% Initial params
sig = 0.01;
sig2 = 100;
nc = 6;
approx = 'eigs'; % 'eign' for Nystrom
nb = 400;

% Construct & plot the dataset
%rng('default')
leng = 1;
for t=1:nb, 
  yin(t,:) = [2.*sin(t/nb*pi*leng) 2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  yang(t,:) = [-2.*sin(t/nb*pi*leng) .45-2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  samplesyin(t,:)  = [yin(t,1)+yin(t,3).*randn   yin(t,2)+yin(t,3).*randn];
  samplesyang(t,:) = [yang(t,1)+yang(t,3).*randn   yang(t,2)+yang(t,3).*randn];
end
h=figure; hold on
plot(samplesyin(:,1),samplesyin(:,2),'o');
plot(samplesyang(:,1),samplesyang(:,2),'o');
xlabel('X_1');
ylabel('X_2');
%title('Structured dataset');
disp('Press any key to continue');
pause;

% Do tuning
Xtr = [samplesyin;samplesyang];
%[lam,U] = kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);
%[lam,U] = kpca(Xtr,'RBF_kernel',sig2);
[lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],approx,nc);
[lam, perm] = sort(-lam); lam = -lam; U = U(:,perm); 
projections = kernel_matrix(Xtr,'RBF_kernel',sig2,Xtr)'*U;
%[Xr,d] = preimage_rbf(Xtr,sig2,U,projections(:,1:nc),'r'); % Reconstruction
%Xd = preimage_rbf(Xtr,sig2,U(:,1:npcs),Xnoisy,'d');      % Denoising
[Xdtr,d] = preimage_rbf(Xtr,sig2,U(:,1:npcs));  % Denoising on the training data

