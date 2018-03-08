clear variables
close all
format compact

%% Load Data
load full_data

sigma=sigma/100;
rf=rf/100;

%% Parameters

rehedge_freq = 1;

%Only used for fallback purpose
callput_scale = 1;
straddle_scale = 1;
hedge_scale = 1;
dynamic_scale = 1;

dynamic_strat_look_back = 120;
%scale_lookback = 30;

is_pure_numerical = true;

date_convension = 360;

%% Initialization

time_since_hedge = 0;
w_mpt = zeros(4, 120+dynamic_strat_look_back);
r_index = zeros(1,size(index,2));
r_ma = zeros(1,size(index,2));
r_callput = zeros(1,size(index,2));
r_straddle = zeros(1,size(index,2));
r_hedge = zeros(1,size(index,2));
r_dynamic = zeros(1,size(index,2));
mpt_decision = zeros(1,size(index,2));
ma_decision = zeros(1,size(index,2));
current_scale = zeros(4,size(index,2));
gamma_callput = zeros(1,size(index,2));
vega_callput = zeros(1,size(index,2));
delta_callput = zeros(1,size(index,2));
rho_callput = zeros(1,size(index,2));
delta_straddle = zeros(1,size(index,2));
gamma_straddle = zeros(1,size(index,2));
vega_straddle = zeros(1,size(index,2));
rho_straddle = zeros(1,size(index,2));
delta_hedge = zeros(1,size(index,2));
gamma_hedge = zeros(1,size(index,2));
vega_hedge = zeros(1,size(index,2));
rho_hedge = zeros(1,size(index,2));

%% Simulation

for i=120:size(index,1)
    %Strategy 4 Hedge Cycle Clock
    T = 90 - mod(i-120,90); %in [1,90]
    
    %MA
    ma60=mean(index(i-59:i));
    ma120=mean(index(i-119:i));
    
    if(i>120)
        %Index
        r_index(i) = ((index(i)/index(i-1)) - 1);
        
        %Strategy 1
        r_ma(i)=(index(i)/index(i-1)-1)*ma_position;
        
        %Strategy 2
        r_callput(i) = (BS(index(i),index(i-1),89/date_convension ,sigma(i), rf(i), is_call)/option - 1);
        
        %Strategy 3
        straddle_today = BS(index(i),index(i-1),89/date_convension ,sigma(i), rf(i), 1)+...
            BS(index(i),index(i-1),89/date_convension ,sigma(i), rf(i), 0);
        straddle_yesterday = (BS(index(i-1),index(i-1), 90/date_convension  ,sigma(i-1), rf(i-1), 1)+...
            BS(index(i-1),index(i-1), 90/date_convension  ,sigma(i-1), rf(i-1), 0));
        r_straddle(i)= (straddle_today/straddle_yesterday - 1);
        
        %Strategy4
        time_since_hedge = time_since_hedge + 1;
        
        if(T == 90)
            T_sell = 0;
        else
            T_sell = T;
        end
        
        today_call = BS(index(i),index(i-time_since_hedge),T_sell/date_convension,sigma(i), rf(i), 1);
        today_put = BS(index(i),index(i-time_since_hedge),T_sell/date_convension ,sigma(i), rf(i), 0);
        today_straddle = today_call + today_put;
        r_hedge(i)= max((today_straddle / hedge_straddle - 1)*hedge_straddle_position,-0.999);
        
        if(time_since_hedge == rehedge_freq || T == 0)
            time_since_hedge = 0;
        end
    end
    
    if ma60>ma120
        %Strategy 1
        ma_position = 1;
        %Strategy 2
        option = BS(index(i),index(i),90/date_convension ,sigma(i), rf(i), 1);
        is_call = 1;
        %Strategy 3
        straddle_position = -1;
        %Strategy 4
        hedge_straddle_position = 1;
        ma_decision(i) = 1;
        
    else
        %Strategy 1
        ma_position = -1;
        %Strategy 2
        option = BS(index(i),index(i), 90/date_convension ,sigma(i), rf(i), 0);
        is_call = 0;
        %Strategy 3
        straddle_position = 1;
        %Strategy 4
        hedge_straddle_position = 1;
        ma_decision(i) = -1;
    end
    %Strategy 4
    hedge_call = BS(index(i),index(i-time_since_hedge),T/date_convension,sigma(i), rf(i), 1);
    hedge_put = BS(index(i),index(i-time_since_hedge),T/date_convension, sigma(i), rf(i), 0);
    hedge_straddle = hedge_call + hedge_put;
    
    
    %Greeks
    [delta_call, delta_put] = blsdelta(index(i), index(i), rf(i), 90/date_convension, sigma(i), 0);
    gamma= blsgamma(index(i), index(i), rf(i), 90/date_convension, sigma(i), 0);
    vega= blsvega(index(i), index(i), rf(i), 90/date_convension, sigma(i), 0);
    [rho_call, rho_put] = blsrho(index(i), index(i), rf(i), 90/date_convension, sigma(i), 0);
    
    %Strategy 2
    gamma_callput(i) = gamma;
    vega_callput(i) = vega;
    if(is_call)
        delta_callput(i) = delta_call;
        rho_callput(i) = rho_call;
    else
        delta_callput(i) = delta_put;
        rho_callput(i) = rho_put;
    end
    
    %Strategy 3
    delta_straddle(i) = delta_call+delta_put;
    gamma_straddle(i) = 2*gamma;
    vega_straddle(i) = 2*vega;
    rho_straddle(i) = rho_call+rho_put;
    
    %Strategy 4
    [delta_call_hedge, delta_put_hedge] = blsdelta(index(i),index(i-time_since_hedge), rf(i), T/date_convension, sigma(i), 0);
    delta_hedge(i) = (delta_call_hedge+delta_put_hedge)*hedge_straddle_position;
    gamma_hedge(i) = (2*blsgamma(index(i),index(i-time_since_hedge), rf(i), T/date_convension, sigma(i), 0))*hedge_straddle_position;
    vega_hedge(i) = (2*blsvega(index(i),index(i-time_since_hedge), rf(i), T/date_convension, sigma(i), 0))*hedge_straddle_position;
    [rho_call_hedge, rho_put_hedge] = blsrho(index(i),index(i-time_since_hedge), rf(i), T/date_convension, sigma(i), 0);
    rho_hedge(i) = (rho_call_hedge+rho_put_hedge)*hedge_straddle_position;
    
    
    %Dynamic Scaling
    [p_call, p_put] = blsprice(index(i),index(i),rf(i),90/date_convension,sigma(i));
    %Strat3
    lambda_call = delta_call*index(i)/p_call;
    lambda_put = delta_put*index(i)/p_put;
    lambda_straddle = p_call/(p_call+p_put)*lambda_call + p_put/(p_call+p_put)*lambda_put;
    d2_straddle = ((rf(i) - 0.5*sigma(i)^2)*90/date_convension)/sqrt(90/date_convension*sigma(i)^2);
    %lambda_straddle = (delta_call+delta_put)*index(i)/(p_call+p_put);
    
    %Strat4
    [p_call_hedge, p_put_hedge] = blsprice(index(i),index(i-time_since_hedge), rf(i), T/date_convension, sigma(i));
    lambda_call_hedge = delta_call_hedge*index(i)/p_call_hedge;
    lambda_put_hedge = delta_put_hedge*index(i)/p_put_hedge;
    lambda_hedge = p_call_hedge/(p_call_hedge+p_put_hedge)*lambda_call_hedge + p_put_hedge/(p_call_hedge+p_put_hedge)*lambda_put_hedge;
    d2_hedge = (log(index(i)/index(i-time_since_hedge)) + (rf(i) - 0.5*sigma(i)^2)*T/date_convension)/sqrt(T/date_convension*sigma(i)^2);
    %lambda_hedge = (delta_call_hedge+delta_put_hedge)*index(i)/(p_call_hedge+p_put_hedge);
    
    current_scale(1,i) = 1;
    current_scale(2,i) = abs(1/delta_callput(i)*option/index(i));
    %current_scale(3,i) = 1/delta_straddle(i)*(p_call+p_put)/index(i);
    %current_scale(4,i) = 1/delta_hedge(i)*(p_call+p_put)/index(i);
    %current_scale(3,i) =  1/lambda_straddle;
    
    current_scale(3,i) = (p_call*lambda_call - p_put*lambda_put)*(p_call+p_put)/(p_call^2*lambda_call^2 + p_put^2*lambda_put^2);
    %current_scale(3,i) = (2*normcdf(d2_straddle)-1)/lambda_straddle;
    %current_scale(3,i) = (d2_straddle*p_call*lambda_call*(p_call + p_put))/(d2_straddle*p_call^2*lambda_call^2 + p_put^2*lambda_put^2);
    
    %current_scale(4,i) =  1/lambda_hedge;
    
    current_scale(4,i) = (p_call_hedge*lambda_call_hedge - p_put_hedge*lambda_put_hedge)...
        *(p_call_hedge+p_put_hedge)/(p_call_hedge^2*lambda_call_hedge^2 + p_put_hedge^2*lambda_put_hedge^2);
    %current_scale(4,i) = (2*normcdf(d2_hedge)-1)/lambda_hedge;
    %current_scale(4,i) = (d2_hedge*p_call_hedge*lambda_call_hedge*(p_call_hedge + p_put_hedge))...
    %    /(d2_hedge*p_call_hedge^2*lambda_call_hedge^2 + p_put_hedge^2*lambda_put_hedge^2);
    
    
%     if(i>120+scale_lookback)
%         idx = i-scale_lookback+1:i;
%         
%         orig_r = [r_ma(idx);r_callput(idx);r_straddle(idx);r_hedge(idx)]';
%         
%         cov_matrix = cov(orig_r);
%         current_scale(:,i) = cov_matrix(:,1)./diag(cov_matrix);
%         %current_scale(:,i) = current_scale(:,i) .* (current_scale(:,i)>0);
%     else
%         current_scale(:,i) = [1;1/20;1/10;1/20];
%     end
    %current_scale(:,i) = [1;1;1;1];
    
    %Strategy Dynamic
    if(i>120+dynamic_strat_look_back+1)
        r_dynamic(i) = [r_ma(i), r_callput(i), r_straddle(i), r_hedge(i)].*current_scale(:,i-1)'*w_mpt(:,i-1);
        %r_dynamic(i) = (1/3*r_index(i) + (2/3*[r_ma(i), r_callput(i), r_straddle(i), r_hedge(i)].*current_scale(:,i-1)'))*w_mpt(:,i-1);
    end
    
%     if(i == 120+dynamic_strat_look_back+1)
%         r_hist = [r_ma;r_callput;r_straddle;r_hedge].*current_scale;
%         r_extreme = [];
%         for j=1:4
%             std_hist = std(r_hist,0,2);
%             extreme_idx = abs(r_hist(j,:))>3*std_hist(j);
%             r_extreme = [r_extreme, r_hist(:,extreme_idx)];
%         end
%         r_extreme = unique(r_extreme,'rows');
%     end
    
    if(i>120+dynamic_strat_look_back)
        window = i-dynamic_strat_look_back+1:i;
        r = [r_ma(window); r_callput(window); r_straddle(window); r_hedge(window)].*current_scale(:,window-1)*date_convension;
        %r = (1/3*r_index(window) + 2/3*[r_ma(window); r_callput(window); r_straddle(window); r_hedge(window)]...
        %    .*current_scale(:,window-1))*date_convension;
        
        %r = r.*(abs(r)<3*std(r,0,2)) + mean(r,2).*ones(size(r)).*(abs(r)>3*std(r,0,2));
        %r = r.*(abs(r)<3*std(r,0,2));
        
%         r = [r_extreme, r];
%         r_hist_std = std([r_ma;r_callput;r_straddle;r_hedge],0,2);
%         if(sum(abs(r(:,end))>r_hist_std)>0)
%             r_extreme = [r_extreme, r(:,end)];
%         end
        
        V = mean(r,2);
        S = cov(r');
        L = ones(size(V));
        A = L'*S^-1*L;
        B = V'*S^-1*L;
        C = V'*S^-1*V;
        D = A*C - B^2;
        Er_min_var = B/A;
        
        %Numerically maximize sharp ratio, s.t. -1<w_i<1
        %func_sharp = @(w) -(w'*V + (1-sum(w))*rf(i) - rf(i))/ sqrt(w'*S*w);
        func_sharp = @(w) -(w'*V - rf(i))/ sqrt(w'*S*w);
        optimizer_option = optimoptions('fmincon','Display', 'off');
        
        if(rf(i) < Er_min_var)
            mpt_decision(i) = 1;
            if(is_pure_numerical)
                w_mpt(:,i) = fmincon(func_sharp,[1;0;0;0], ones(1,4), 1, [],[], ones(4,1)*-1, ones(4,1)*1, [], optimizer_option);
                
            else
                %Use MPT formula
                w_mpt(:,i) = (S^-1 *(V - rf(i)*L))/(B - A*rf(i));
            end
        else
            mpt_decision(i) = -1;
            %Numerically maximize sharp ratio, s.t. -1<w_i<1
            %disp(i)
            w_mpt(:,i) = fmincon(func_sharp,[1;0;0;0], ones(1,4), 1, [],[], ones(4,1)*-1, ones(4,1)*1, [], optimizer_option);
            
            %Reverse MPT formula
            %w_mpt(:,i) = -(S^-1 *(V - rf(i)*L))/(B - A*rf(i));
            
%             if(w_mpt(:,i)'*V < rf(i))
%                 w_mpt(:,i) = 0;
%                 disp(i);
%             end

        end

        %Use the best strategy without mixing
        %w_mpt(:,i) = w_mpt(:,i).*(w_mpt(:,i) == max(w_mpt(:,i)));

    end
    
    
end

%% Result Preprocessing

start_idx = 120+dynamic_strat_look_back+1;

sample_index = start_idx:size(r_index,2);
%sample_index = start_idx+2240:size(r_index,2);

% without scaling.
% r_matrix = [r_index; ...
%     r_ma;...
%     r_callput;...
%     r_straddle;...
%     r_hedge;...
%     r_dynamic];

% combine 1/3 and 2/3
% r_matrix(1:5,:) = 2/3*r_matrix(1:5,:) + 1/3*r_index(sample_index);


% dynamic scaling
r_matrix = [r_index(sample_index); ...
    r_ma(sample_index).*current_scale(1,sample_index-1) + rf(sample_index)'.*(1-current_scale(1,sample_index-1))/date_convension;...
    r_callput(sample_index).*current_scale(2,sample_index-1) + rf(sample_index)'.*(1-current_scale(2,sample_index-1))/date_convension;
    r_straddle(sample_index).*current_scale(3,sample_index-1) + rf(sample_index)'.*(1-current_scale(3,sample_index-1))/date_convension;
    r_hedge(sample_index).*current_scale(4,sample_index-1) + rf(sample_index)'.*(1-current_scale(4,sample_index-1))/date_convension;...
    r_dynamic(sample_index)*dynamic_scale + rf(sample_index)'.*(1-sum(w_mpt(:,sample_index-1)))/date_convension];

% r_matrix = [r_index(sample_index); ...
%     r_ma(sample_index).*current_scale(1,sample_index-1);...
%     r_callput(sample_index).*current_scale(2,sample_index-1);
%     r_straddle(sample_index).*current_scale(3,sample_index-1);
%     r_hedge(sample_index).*current_scale(4,sample_index-1);...
%     r_dynamic(sample_index)*dynamic_scale];







%% Sharp Ratio

r_mean = mean(r_matrix - rf(sample_index)'/date_convension,2)*date_convension;

r_std = std(r_matrix,0,2)*sqrt(date_convension);

r_sharpratio = r_mean./r_std;


%% VaR and CVaR

%VaR, CVaR for original portfolio (historical method)

sorted_r_matrix = sort(r_matrix,2);

significance = 0.05;
quantile = round(significance * size(r_matrix,2));

VaR = sorted_r_matrix(:,quantile);
CVaR = mean(sorted_r_matrix(:,1:quantile),2);


[MaxDD, MaxDDIndex] = maxdrawdown(cumprod(r_matrix+1,2)');


%% Risk Prediction

var_window = 90;

moving_var_index = var(createRollingWindow(r_matrix(1,:) - rf(sample_index)'/date_convension, var_window),0,2);
moving_var_ma = var(createRollingWindow(r_matrix(2,:) - rf(sample_index)'/date_convension, var_window),0,2);
moving_var_callput = var(createRollingWindow(r_matrix(3,:)*callput_scale - rf(sample_index)'/date_convension, var_window),0,2);
moving_var_straddle = var(createRollingWindow(r_matrix(4,:)*straddle_scale - rf(sample_index)'/date_convension, var_window),0,2);
moving_var_hedge = var(createRollingWindow(r_matrix(5,:)*hedge_scale - rf(sample_index)'/date_convension, var_window),0,2);
moving_var_dynamic = var(createRollingWindow(r_matrix(6,:)*dynamic_scale - rf(sample_index)'/date_convension, var_window),0,2);

% figure;
% hold on;
% plot(moving_var_index);
% plot(moving_var_ma);
% plot(moving_var_callput);
% plot(moving_var_straddle);
% plot(moving_var_hedge);
% plot(moving_var_dynamic);
% legend('index', 'strat1', 'strat2', 'strat3', 'strat4', 'dynamic');



%% Visualization


% ma_index_60 = [index(1:59)', mean(createRollingWindow(index,60),2)'];
% ma_index_120 = [index(1:119)', mean(createRollingWindow(index,120),2)'];
% diff =   ma_index_60 - ma_index_120;
% ma_diff = [zeros(1,11),  mean(createRollingWindow(diff,12),2)'];
% macd = diff - ma_diff;
% figure;
% hold on;
% plot(index(1500:2500));
% plot(ma_index_60(1500:2500));
% plot(ma_index_120(1500:2500));
% plot(macd(1500:2500)*10 + index(1500));
% plot(ones(1,1001).*index(1500));

wealth = cumprod(r_matrix+1,2);
disp('sharp,    wealth,     MaxDD,      VaR,       CVaR');
disp([r_sharpratio,wealth(:,end),MaxDD', VaR, CVaR]);


figure;
hold on;
plot(sample_index,wealth(:,:));
%plot(Date(sample_index), wealth);
% plot(Date(sample_index), cumprod(r_index + 1));
% plot(Date(sample_index),cumprod(r_ma + 1));
% plot(Date(sample_index),cumprod(r_callput*callput_scale + 1));
% plot(Date(sample_index),cumprod(r_straddle*straddle_scale + 1));
% plot(Date(sample_index),cumprod(r_hedge*hedge_scale + 1));
% plot(Date(sample_index),cumprod(r_dynamic*dynamic_scale + 1));
legend('index', 'strat1', 'strat2', 'strat3', 'strat4', 'dynamic');
hold off;

% 
% figure;
% subplot(2,2,1);
% hold on;
% histogram(r_matrix(2,:));
% histogram(r_matrix(3,:));
% legend('strat1','strat2');
% subplot(2,2,2);
% hold on;
% histogram(r_matrix(2,:));
% histogram(r_matrix(4,:));
% legend('strat1','strat3');
% subplot(2,2,3);
% hold on;
% histogram(r_matrix(2,:));
% histogram(r_matrix(5,:));
% legend('strat1','strat4');
% subplot(2,2,4);
% hold on;
% histogram(r_matrix(2,:));
% histogram(r_matrix(6,:));
% legend('strat1','dynamic');
% hold off;
% 
% 
% figure;
% ma_plot = ma_decision.*[0, ma_decision(2:end) ~= ma_decision(1:end-1)];
% mpt_plot = mpt_decision.*[0, mpt_decision(2:end) ~= mpt_decision(1:end-1)];
% 
% %subplot(2,1,1);
% plot(Date(sample_index), cumprod(r_matrix(1,:) + 1));
% %subplot(2,1,2);
% hold on;
% plot(Date(sample_index),ma_plot(sample_index)+1);
% %plot(Date(sample_index),mpt_plot(sample_index));
% hold off;
% 
% 
% %Plot Greeks
figure;
subplot(2,2,1);
hold on;
plot(sample_index,delta_callput(sample_index));
plot(sample_index,delta_straddle(sample_index));
plot(sample_index,delta_hedge(sample_index));
legend('delta callput', 'delta straddle', 'delta hedge');


subplot(2,2,2);
hold on;
plot(Date(sample_index), gamma_callput(sample_index));
plot(Date(sample_index),gamma_straddle(sample_index));
plot(Date(sample_index),gamma_hedge(sample_index));
legend('gamma callput', 'gamma straddle', 'gamma hedge');


subplot(2,2,3);
hold on;
plot(Date(sample_index), vega_callput(sample_index));
plot(Date(sample_index),vega_straddle(sample_index));
plot(Date(sample_index),vega_hedge(sample_index));
legend('vega callput', 'vega straddle', 'vega hedge');

subplot(2,2,4);
hold on;
plot(Date(sample_index), rho_callput(sample_index));
plot(Date(sample_index),rho_straddle(sample_index));
plot(Date(sample_index),rho_hedge(sample_index));
legend('rho callput', 'rho straddle', 'rho hedge');
hold off;

%% stop loss from HTC

clear all
close all
load final_v3_result_no_combine

cumr_matrix=cumprod(r_matrix+1,2);
boundary_down=0.10;
boundary_up=0.10;
r_matrixnew=r_matrix;
cumr_tran=-cumr_matrix+max(cumr_matrix)+0.1;
rf_new=rf(sample_index);
pointdown=[1,1,1,1,1];
pointup=[1,1,1,1,1];
for i=3:size(cumr_matrix,2)
    for k=1:size(pointdown,2)
        [r_maxdd(i,k),r_maxddindex]=maxdrawdown(cumr_matrix(k,pointdown(k):i-1)');
        [r_maxdu(i,k),r_maxduindex]=maxdrawdown(cumr_tran(k,pointup(k):i-1)');
    end
    for j=1:size(cumr_matrix,1)-1
        if r_maxdd(i-1,j)>=boundary_down  && r_matrixnew(j,i-2)~=rf_new(i-2)/360
           pointup(j)=i-1;
        end
        if r_maxdd(i,j)>=boundary_down 
           r_matrixnew(j,i)=rf_new(i)/360;
        end
        if r_maxdu(i-1,j)>=boundary_up && r_matrixnew(j,i-2)==rf_new(i-2)/360
            pointdown(j)=i-1;
        end
    end
end
r_meannew = mean(r_matrixnew - rf(sample_index)'/date_convension,2)*date_convension;
r_stdnew = std(r_matrixnew,0,2)*sqrt(date_convension);
r_sharprationew = r_meannew./r_stdnew;
figure;
hold on;
plot(sample_index,cumprod(r_matrixnew+1,2));
%plot(sample_index,cumprod(r_matrix+1,2));
hold off;


%% stop loss from CYZ
cumr_matrix=cumprod(r_matrix+1,2);
boundary_down=0.10;
boundary_up=0.10;
r_matrixnew=r_matrix;
cumr_tran=-cumr_matrix+max(cumr_matrix)+0.1;
rf_new=rf(sample_index);
pointdown=[1,1,1,1,1];
pointup=[1,1,1,1,1];
for i=3:size(cumr_matrix,2)
    for k=1:size(pointdown,2)
        [r_maxdd(i,k),r_maxddindex]=maxdrawdown(cumr_matrix(k,pointdown(k):i-1)');
        [r_maxdu(i,k),r_maxduindex]=maxdrawdown(cumr_tran(k,pointup(k):i-1)');
    end
    for j=1:size(cumr_matrix,1)-1
        if r_maxdd(i-1,j)>=boundary_down  && r_matrixnew(j,i-2)~=rf_new(i-2)/360
           pointup(j)=i-1;
        end
        if r_maxdd(i,j)>=boundary_down 
           r_matrixnew(j,i)=rf_new(i)/360;
        end
        if r_maxdu(i-1,j)>=boundary_up && r_matrixnew(j,i-2)==rf_new(i-2)/360
            pointdown(j)=i-1;
        end
    end
end
r_meannew = mean(r_matrixnew - rf(sample_index)'/date_convension,2)*date_convension;
r_stdnew = std(r_matrixnew,0,2)*sqrt(date_convension);
r_sharprationew = r_meannew./r_stdnew;

%%we find that the hedge strategy under this stop loss strategy perform the
%%the best.
figure;
hold on;
plot(Date(sample_index),cumprod(r_matrix(1,:)+1,2));
plot(Date(sample_index),cumprod(r_matrix(2,:)+1,2));
plot(Date(sample_index),cumprod(r_matrixnew(2,:)+1,2));
legend('S&P500 Index', 'Portfolio 1', 'Portfolio 1 (stop loss)');
xlabel('Date')
ylabel('Cumulated Return')
hold off;

% new sharp, wealth, MaxDD var,cvar under stop loss strategy
r_meannew = mean(r_matrixnew - rf(sample_index)'/date_convension,2)*date_convension;

r_stdnew = std(r_matrixnew,0,2)*sqrt(date_convension);

r_sharprationew = r_meannew./r_stdnew;

sorted_r_matrixnew = sort(r_matrixnew,2);

significance = 0.05;
quantilenew = round(significance * size(r_matrixnew,2));

VaRnew = sorted_r_matrixnew(:,quantile);
CVaRnew = mean(sorted_r_matrixnew(:,1:quantile),2);


[MaxDDnew, MaxDDIndexnew] = maxdrawdown(cumprod(r_matrixnew+1,2)');

wealthnew = cumprod(r_matrixnew+1,2);
disp('sharp,    wealth,     MaxDD');
disp([r_sharprationew,wealthnew(:,end),MaxDDnew']);

%% other figure

% Scenario Partition from bessie
first = datenum('03-Jan-2007','dd-mmm-yyyy');
figure(5);
hold on;
%plot(sample_index,cumprod(r_matrix+1,2));
a=patch([first first+308 first+308 first], [0 0 3 3],[0.25, 0.25, 0.25]);
b=patch([first+309 first+1241 first+1241 first+309], [0 0 3 3],[0.25, 0.25, 0.25]);
c=patch([first+1242 first+1911 first+1911 first+1242], [0 0 3 3],[0.25, 0.25, 0.25]);
d=patch([first+1912 first+2240 first+2240 first+1912], [0 0 3 3],[0.25, 0.25, 0.25]);
e=patch([first+2241 first+2529 first+2529 first+2241], [0 0 3 3],[0.25, 0.25, 0.25]);
axis([first first+2529 0 3])
% a.EdgeColor = 'w';
% b.EdgeColor = 'w';
% c.EdgeColor = 'w';
% d.EdgeColor = 'w';
% e.EdgeColor = 'w';
alpha(a,0.5); 
alpha(b,0.3);
alpha(c,0.1); 
alpha(d,0.3); 
alpha(e,0.1); 
textname = [1,2,3,4,5];
x = [first+80,first+660,first+1450,first+2000,first+2300];
y = [0.25,0.25,0.25,0.25,0.25];
for k = 1:5    
        text(x(k), y(k), sprintf('Period %d', textname(k)))
end
plot(first:first+2528,cumprod(r_matrix(1:5,:)+1,2),'LineWidth',2);
datetick('x', 'mm/dd/yyyy')
hold off;

%% other figures 2
%figure from sara and mengrui.

close all
clear all
load final_v3_result_no_combine.mat

%Bear Market Comparison (01/03/2008 - 03/06/2009)
figure(1); 
subplot(211);
hold on;
plot(Date(sample_index(13:308)), cumprod(r_matrix(1:3,13:308)+1,2)');
legend('S&P 500 Index', 'Portfolio 1', 'Portfolio 2');
xlabel('Time');
ylabel('Cumulative Return');
subplot(212);
hold on;
plot(Date(sample_index(13:308)), cumprod(r_matrix([1,4:5],13:308)+1,2)');
legend('S&P 500 Index', 'Portfolio 3', 'Portfolio 4');
xlabel('Time');
ylabel('Cumulative Return');

%Bull Market Comparison (12/31/2012 - 10/14/2014)
figure(2); 
subplot(211);
hold on;
plot(Date(sample_index(1270:1720)), cumprod(r_matrix(1:3,1270:1720)+1,2)');
legend('S&P 500 Index', 'Portfolio 1', 'Portfolio 2');
xlabel('Time');
ylabel('Cumulative Return');
xlim([Date(sample_index(1270)),Date(sample_index(1720))])
subplot(212);
hold on;
plot(Date(sample_index(1270:1720)), cumprod(r_matrix([1,4:5],1270:1720)+1,2)');
legend('S&P 500 Index', 'Portfolio 3', 'Portfolio 4');
xlabel('Time');
ylabel('Cumulative Return');
xlim([Date(sample_index(1270)),Date(sample_index(1720))])


% w_mpt plotting %Positions in Strategy 1&2 and 3&4
w_mpt1 = zeros(4, length(w_mpt(1,241:end)));
for j = 1:length(w_mpt(1, 241:end))
    if w_mpt(3,j+240) > w_mpt(4,j+240)
        w_mpt1(3,j) = 1;
        w_mpt1(4,j) = -1;
    
    else 
        w_mpt1(3,j) = -1;
        w_mpt1(4,j) = 1;
    end
    if w_mpt(1,j+240) > w_mpt(2,j+240)
        w_mpt1(1,j) = 1;
        w_mpt1(2,j) = -1;
    
    else 
        w_mpt1(1,j) = -1;
        w_mpt1(2,j) = 1;
    end
end

figure(3);  
subplot(2,1,1);
hold on;
plot(Date(sample_index),index(sample_index)'/600-2.5);
plot(Date(sample_index),w_mpt1(1:2,:)');
legend('S&P500 Index','Portfolio 1', 'Portfolio 2');
%axis([0 length(w_mpt1(1,:)) -1.5 1.5]);
hold off; 

subplot(2,1,2);
hold on;
plot(Date(sample_index), index(sample_index)'/600-2.5);
plot(Date(sample_index),w_mpt1(3:4,:)');
hold off;
legend('S&P500 Index','Portfolio 3', 'Portfolio 4');
%axis([0 2529 -1.5 1.5]);

% Daily return(90 days) %Daily Return of Strategy 1&2 and 3&4
figure(4);  
subplot(2,1,1);
hold on; 
plot(Date(sample_index(61:150)), r_matrix(2:3,61:150)'); 
legend('Portfolio 1', 'Portfolio 2'); 
xlabel('Date'); 
ylabel('Daily Return'); 
hold off;

subplot(2,1,2);
hold on; 
plot(Date(sample_index(61:150)), r_matrix(4:5,61:150)'); 
legend('Portfolio 3', 'Portfolio 4'); 
xlabel('Date'); 
ylabel('Daily Return'); 
hold off;



f = plot(first:first+2528,cumprod(r_matrix(1:5,:)+1,2),'LineWidth',1.5);
xticks([first+308 first+1241 first+1911 first+2240 first+2529]); % Change x-axis ticks
xticklabels({'03/2008','12/2011','08/2014','11/2015','01/2017'});
legend(f,{'S&P500 Index', 'Portfolio 1', 'Portfolio 2', 'Portfolio 3', 'Portfolio 4'});
xlabel('Date')
ylabel('Cumulated Return')
hold off;

%% abandon 


% Risk Prediction

var_window = 90;

moving_var_index = var(createRollingWindow(r_index(sample_index) - rf(sample_index)'/360, var_window),0,2);
moving_var_ma = var(createRollingWindow(r_ma(sample_index) - rf(sample_index)'/360, var_window),0,2);
moving_var_callput = var(createRollingWindow(r_callput(sample_index)*callput_scale - rf(sample_index)'/360, var_window),0,2);
moving_var_straddle = var(createRollingWindow(r_straddle(sample_index)*straddle_scale - rf(sample_index)'/360, var_window),0,2);
moving_var_hedge = var(createRollingWindow(r_hedge(sample_index)*hedge_scale - rf(sample_index)'/360, var_window),0,2);
moving_var_dynamic = var(createRollingWindow(r_dynamic(sample_index)*dynamic_scale - rf(sample_index)'/360, var_window),0,2);

figure;
hold on;
plot(moving_var_index);
plot(moving_var_ma);
plot(moving_var_callput);
plot(moving_var_straddle);
plot(moving_var_hedge);
plot(moving_var_dynamic);
legend('index', 'strat1', 'strat2', 'strat3', 'strat4', 'dynamic');

%% abandon 
% GARCH fitting
%find proper lags
%fitdata = round(length(r_ma) * 0.8);
%testdata = length(r_ma) - fitdata;

fitdata = length(r_ma);

AIC_ma = 10000;
AIC_callput = 10000;
AIC_straddle = 10000;
AIC_hedge = 10000;
AIC_dynamic = 10000;
lag_ma = zeros(1,2);
lag_callput = zeros(1,2);
lag_straddle = zeros(1,2);
lag_hedge = zeros(1,2);
lag_dynamic = zeros(1,2);



for p = 1:3
    for q = 1:3
         func1 = garch('GARCHLags',p,'ARCHLags',q,'Offset',NaN);
         [modl_ma, paramcov_ma, ll_ma, info1] = estimate(func1, r_ma(sample_index)'-rf(sample_index)/360);
         [modl_callput, paramcov_callput, ll_callput, info2] = estimate(func1, r_callput(sample_index)'*callput_scale-rf(sample_index)/360);
         [modl_straddle, paramcov_straddle, ll_straddle, info3] = estimate(func1, r_straddle(sample_index)'*straddle_scale-rf(sample_index)/360);
         [modl_hedge, paramcov_hedge,ll_hedge, info4] = estimate(func1, r_hedge(sample_index)'*hedge_scale-rf(sample_index)/360);
         [modl_dynamic, paramcov_dynamic,ll_dynamic, info5] = estimate(func1, r_dynamic(sample_index)'*hedge_scale-rf(sample_index)/360);
         
         aic_ma = aicbic(ll_ma, sum(any(paramcov_ma)));
         aic_callput = aicbic(ll_callput, sum(any(paramcov_callput)));
         aic_straddle = aicbic(ll_straddle, sum(any(paramcov_straddle)));
         aic_hedge = aicbic(ll_hedge, sum(any(paramcov_hedge)));
         aic_dynamic = aicbic(ll_dynamic, sum(any(paramcov_dynamic)));
         if aic_ma < AIC_ma
             lag_ma = [p, q];
             AIC_ma = aic_ma;
         end
         if aic_callput < AIC_callput
             lag_callput = [p, q];
             AIC_callput = aic_callput;
         end
         if aic_straddle < AIC_straddle
             lag_straddle = [p, q];
             AIC_straddle = aic_straddle;
         end
         if aic_hedge < AIC_hedge
             lag_hedge = [p, q];
             AIC_hedge = aic_hedge;
         end
         if aic_dynamic < AIC_dynamic
             lag_dynamic = [p, q];
             AIC_dynamic = aic_dynamic;
         end
    end
end


% Build model 
func_ma = garch('GARCHLags', lag_ma(1), 'ARCHLags', lag_ma(2), 'Offset',NaN);
func_callput = garch('GARCHLags', lag_callput(1), 'ARCHLags', lag_callput(2), 'Offset',NaN);
func_straddle = garch('GARCHLags', lag_straddle(1), 'ARCHLags', lag_straddle(2), 'Offset',NaN);
func_hedge = garch('GARCHLags', lag_hedge(1), 'ARCHLags', lag_hedge(2), 'Offset',NaN);
func_dynamic = garch('GARCHLags', lag_dynamic(1), 'ARCHLags', lag_dynamic(2), 'Offset',NaN);
mod_ma = estimate(func_ma, r_ma(sample_index)'-rf(sample_index)/360);
mod_callput = estimate(func_callput, r_callput(sample_index)'*callput_scale-rf(sample_index)/360);
mod_straddle = estimate(func_straddle, r_straddle(sample_index)'*straddle_scale-rf(sample_index)/360);
mod_hedge = estimate(func_hedge, r_hedge(sample_index)'*hedge_scale-rf(sample_index)/360);
mod_dynamic = estimate(func_dynamic, r_dynamic(sample_index)'*hedge_scale-rf(sample_index)/360);


% infer
% forecast
% mod_ma;
% aa = ones(3,1);
% aa2 = infer(mod_ma, aa);

%infer
infer_ma = infer(mod_ma, r_ma(sample_index)'-rf(sample_index)/360);
infer_callput = infer(mod_callput, r_callput(sample_index)'*callput_scale-rf(sample_index)/360);
infer_straddle = infer(mod_straddle, r_straddle(sample_index)'*straddle_scale-rf(sample_index)/360);
infer_hedge = infer(mod_hedge, r_hedge(sample_index)'*hedge_scale-rf(sample_index)/360);
infer_dynamic = infer(mod_dynamic, r_dynamic(sample_index)'*hedge_scale-rf(sample_index)/360);

%fitness: MSE

fit_infer_ma = goodnessOfFit(infer_ma(var_window-1:length(infer_ma)-1), moving_var_ma, 'MSE');
fit_infer_callput = goodnessOfFit(infer_callput(var_window-1:length(infer_ma)-1), moving_var_callput, 'MSE');
fit_infer_straddle = goodnessOfFit(infer_straddle(var_window-1:length(infer_ma)-1), moving_var_straddle, 'MSE');
fit_infer_hedge = goodnessOfFit(infer_hedge(var_window-1:length(infer_ma)-1), moving_var_hedge, 'MSE');
fit_infer_dynamic = goodnessOfFit(infer_dynamic(var_window-1:length(infer_ma)-1), moving_var_dynamic, 'MSE');

%R square
mdl1_infer_sVar = fitlm(infer_ma(var_window:end),moving_var_ma);
mdl2_infer_sVar = fitlm(infer_callput(var_window:end),moving_var_callput);
mdl3_infer_sVar = fitlm(infer_straddle(var_window:end), moving_var_straddle);
mdl4_infer_sVar = fitlm(infer_hedge(var_window:end), moving_var_hedge);
mdl5_infer_sVar = fitlm(infer_dynamic(var_window:end), moving_var_dynamic);

Rsquare_MA_infer_sVar = mdl1_infer_sVar.Rsquared.Adjusted;
Rsquare_MACP_infer_sVar = mdl2_infer_sVar.Rsquared.Adjusted;
Rsquare_MAST_infer_sVar = mdl3_infer_sVar.Rsquared.Adjusted;
Rsquare_MAHE_infer_sVar = mdl4_infer_sVar.Rsquared.Adjusted;
Rsquare_MADY_infer_sVar = mdl5_infer_sVar.Rsquared.Adjusted;

%plot
figure;
hold on;
plot(infer_ma(var_window:end));
plot(infer_callput(var_window:end));
plot(infer_straddle(var_window:end));
plot(infer_hedge(var_window:end));
plot(infer_dynamic(var_window:end));
legend('ma', 'cp', 'st', 'hegde', 'dynamic');


%plot combine
figure;
hold on;
plot(infer_ma(var_window:end));
plot(infer_callput(var_window:end));
plot(infer_straddle(var_window:end));
plot(infer_hedge(var_window:end));
plot(infer_dynamic(var_window:end));
plot(moving_var_index);
plot(moving_var_ma);
plot(moving_var_callput);
plot(moving_var_straddle);
plot(moving_var_hedge);
plot(moving_var_dynamic);
legend('infer_ma', 'infer_callput', 'infer_straddle', 'infer_hedge', 'infer_dynamic', 'moving_var_index','moving_var_ma', 'moving_var_callput', 'moving_var_straddle', 'moving_var_hedge', 'moving_var_dynamic');


% simulate
nsim = 1000;
[simvar_ma, simr_ma] = simulate(mod_ma, fitdata, 'Numpaths', nsim);
[simvar_callput, simr_callput] = simulate(mod_callput, fitdata, 'Numpaths', nsim);
[simvar_straddle, simr_straddle] = simulate(mod_straddle, fitdata, 'Numpaths', nsim);
[simvar_hedge, simr_hedge] = simulate(mod_hedge, fitdata, 'Numpaths', nsim);
[simvar_dynamic, simr_dynamic] = simulate(mod_dynamic, fitdata, 'Numpaths', nsim);

svar_ma = mean(simvar_ma,2);
svar_callput = mean(simvar_callput,2);
svar_straddle = mean(simvar_straddle,2);
svar_hedge = mean(simvar_hedge,2);
svar_dynamic = mean(simvar_dynamic,2);


%plot
figure;
hold on;
plot(svar_ma(var_window:end));
plot(svar_callput(var_window:end));
plot(svar_straddle(var_window:end));
plot(svar_hedge(var_window:end));
plot(svar_dynamic(var_window:end));
legend('strat1', 'strat2', 'strat3', 'strat4', 'dynamic');

%plot combine
figure;
hold on;
plot(svar_ma(var_window:end));
plot(svar_callput(var_window:end));
plot(svar_straddle(var_window:end));
plot(svar_hedge(var_window:end));
plot(svar_dynamic(var_window:end));
plot(moving_var_index);
plot(moving_var_ma);
plot(moving_var_callput);
plot(moving_var_straddle);
plot(moving_var_hedge);
plot(moving_var_dynamic);
legend('svar_ma', 'svar_callput', 'svar_straddle', 'svar_hedge', 'svar_dynamic', 'moving_var_index','moving_var_ma', 'moving_var_callput', 'moving_var_straddle', 'moving_var_hedge', 'moving_var_dynamic');

%fitness: MSE
fit_ma = goodnessOfFit(svar_ma(var_window-1:length(infer_ma)-1), moving_var_ma, 'MSE');
fit_callput = goodnessOfFit(svar_callput(var_window-1:length(infer_ma)-1), moving_var_callput, 'MSE');
fit_straddle = goodnessOfFit(svar_straddle(var_window-1:length(infer_ma)-1), moving_var_straddle, 'MSE');
fit_hedge = goodnessOfFit(svar_hedge(var_window-1:length(infer_ma)-1), moving_var_hedge, 'MSE');
fit_dynamic = goodnessOfFit(svar_dynamic(var_window-1:length(infer_ma)-1), moving_var_dynamic, 'MSE');

%R square
mdl1_sVar = fitlm(svar_ma(var_window:end),moving_var_ma);
mdl2_sVar = fitlm(svar_callput(var_window:end),moving_var_callput);
mdl3_sVar = fitlm(svar_straddle(var_window:end), moving_var_straddle);
mdl4_sVar = fitlm(svar_hedge(var_window:end), moving_var_hedge);
mdl5_sVar = fitlm(svar_dynamic(var_window:end), moving_var_dynamic);

Rsquare_MA_sVar = mdl1_sVar.Rsquared.Adjusted;
Rsquare_MACP_sVar = mdl2_sVar.Rsquared.Adjusted;
Rsquare_MAST_sVar = mdl3_sVar.Rsquared.Adjusted;
Rsquare_MAHE_sVar = mdl4_sVar.Rsquared.Adjusted;
Rsquare_MADY_sVar = mdl5_sVar.Rsquared.Adjusted;


%% Support Functions

function option = BS(S,K,tau,sigma,r, is_call)

d1 = (log(S/K) + (r + 0.5*sigma^2)*tau)/sqrt(tau*sigma^2);
d2 = (log(S/K) + (r - 0.5*sigma^2)*tau)/sqrt(tau*sigma^2);

if(is_call)
    option =  S*normcdf(d1) - K*exp(-r*tau)*normcdf(d2);
else
    option = K*exp(-r*tau)*normcdf(-d2) - S*normcdf(-d1);
end

end

function output = createRollingWindow(vector, n)
    % CREATEROLLINGWINDOW returns successive overlapping windows onto a vector
    %   OUTPUT = CREATEROLLINGWINDOW(VECTOR, N) takes a numerical vector VECTOR
    %   and a positive integer scalar N. The result OUTPUT is an MxN matrix,
    %   where M = length(VECTOR)-N+1. The I'th row of OUTPUT contains
    %   VECTOR(I:I+N-1).
    l = length(vector);
    m = l - n + 1;
    output = vector(hankel(1:m, m:l));
end

