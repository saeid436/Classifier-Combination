%% Block Mean Feature Extraction Function:

function [FeatureVec] = blockMeanFeatures(Image,H,W,WinSize)

	ImageResize = imresize(Image,[H W]); % Resize Input Image TO Desired Scale
	FeatureVec = zeros(((H/WinSize)*(W/WinSize))+1,1); % A Vector For Save Features
	FeatureVec(1) = 1; % Bias
	K = 2;
  
	for i = 1 : WinSize : H
		for j = 1 : WinSize : W
			BlockAvarage = sum(sum(ImageResize(i : i+WinSize-1,j : j+WinSize-1)))/(WinSize^2); % Compute Values Of Means For Every Block
			FeatureVec(K) = BlockAvarage;
			K = K+1;
		end
	end
	
  MaxF = max(FeatureVec(2:end));
	MinF = min(FeatureVec(2:end));
	FeatureVec(2:end) = (FeatureVec(2:end)-MinF)/(MaxF-MinF);

end