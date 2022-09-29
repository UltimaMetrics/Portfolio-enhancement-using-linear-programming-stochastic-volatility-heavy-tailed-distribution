% % =======================================================================
% % Stochastic volatility model with a constant mean
% %
% % y_t = mu + epsilon_t,                    epsilon_t ~ N(0,exp(h_t)),
% % h_t = muh + phih(h_{t-1}-muh) + zeta_t,  zeta_t ~ N(0,sigh2),
% % 
% % See Chan, J.C.C. and Hsiao, C.Y.L (2014). Estimation of Stochastic
% % Volatility Models with Heavy Tails and Serial Dependence. 
% % In: I. Jeliazkov and X.S. Yang (Eds.), Bayesian Inference in the 
% % Social Sciences, 159-180, John Wiley & Sons, New York.
% %
% % (c) 2013, Joshua Chan. Email: joshuacc.chan@gmail.com
% % =======================================================================

clear; clc;
nloop = 5000;
burnin = 1000;
load 'TW50.csv';
y = TW50; 
T = length(y);

%% prior
invVmu = 1/5;
phih0 = .95; invVphih = 1;
muh0 = 0; invVmuh = 1/5;
nuh = 10; Sh = .02*(nuh-1);

disp('Starting MCMC.... ');
disp(' ' );
start_time = clock;    
    
% initialize the Markov chain
sigh2 = .05;
phih = .95;
muh = 1;
mu = mean(y);
h = log(var(y)*.8)*ones(T,1);

% initialize for storage
store_theta = zeros(nloop - burnin,4); % [mu muh phih sigh2]
store_exph = zeros(nloop - burnin,T);  % store exp(h_t/2)

%% compute a few things outside the loop
newnuh = T/2 + nuh;
rand('state', sum(100*clock) ); randn('state', sum(200*clock) );

for loop = 1:nloop
        %% sample mu    
    invexph = exp(-h);
    Dmu = 1/(invVmu + sum(invexph));
    muhat = Dmu*sum(invexph.*y);
    mu = muhat + sqrt(Dmu)*randn;        
        %% sample h, muh, phih, sigh2
    Ystar = log((y-mu).^2 + .0001);
    [h muh phih sigh2] = SV(Ystar,h,muh,phih,sigh2,[muh0 invVmuh ...
        phih0 invVphih nuh Sh]);    
    if ( mod( loop, 2000 ) ==0 )
        disp(  [ num2str( loop ) ' loops... ' ] )
    end    
    if loop>burnin
        i = loop-burnin;
        store_exph(i,:) = exp(h/2)'; 
        store_theta(i,:) = [mu muh phih sigh2];
    end    
end

disp( ['MCMC takes '  num2str( etime( clock, start_time) ) ' seconds' ] );
disp(' ' );

thetahat = mean(store_theta)';
exphhat = mean(store_exph)'; 
exphlb = quantile(store_exph,.05)';
exphub = quantile(store_exph,.95)';
figure;
tid = linspace(2011,2021,T)';
figure; plot(tid, [exphhat exphlb exphub]);
box off; xlim([2011 2021]);


