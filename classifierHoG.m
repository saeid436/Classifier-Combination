%% Training Classifier Based on HoG Features:

function [Wih, Who] = classifierHoG(TrainSample,TrainTarget,TrainCont,Alpha,NH1,Epoch,m)
	
	%% Training Section For Weights Of HoG Features
	
	Wih = (rand(182,NH1)-.5)*.5; % Weights Between Inputs And Hidden Layer
	Who = (rand(NH1+1,m)-.5)*.5; % Weights Between Hidden Layer And Out Put
	
	for epoch = 1:Epoch
		for r = 1:TrainCont
					
			nEt= Wih' * TrainSample(:,r);
			Z= 1./(1+exp(-nEt));  %output for the hidden layer 
			Zb=[1;Z]; %adding a bias
			
			nEt2= Who' * Zb ;
			Output= 1./(1+exp(-nEt2));  %Final outputs
			
			
			% Adjust delta values of weights For output layer:
	        delta_o = Output.*(1-Output).*(TrainTarget(:,r)-Output);
			
			% Propagate the delta backwards into hidden layers:	     
	        ezb= Who * delta_o;
	        delta_hb = Zb.*(1-Zb).*ezb;
	        delta_h=delta_hb(2:end);
			% Add weight changes to original weights:
			
	        Who = Who + Alpha*Zb*delta_o';       
	        Wih= Wih + Alpha*TrainSample(:,r)*delta_h';
			
		end
	end
end