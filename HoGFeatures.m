%% Histogram of Oriented Gradient(HoG) Feature Extraction Fucntion:

function HoG = HoGFeatures(Image,H,W)
Image = imresize(Image,[H,W]);
Image = double(Image);
[Gx,Gy] = imgradientxy(Image);
AmpG = sqrt(Gx.^2 + Gy.^2);
Theta = atand(Gy./(Gx+eps));
Theta = Theta + 90;
Theta = round(Theta);
% if(Theta == NaN)
%     Theta = 0;
% else
%     Theta = Theta;
% end
Histogram = zeros(1,181);
for i = 1:H
    for j = 1:W
        Angle = Theta(i,j);
        A = AmpG(i,j);
        Histogram(Angle+1) = Histogram(Angle+1) + A;
    end
end
MaxHist = max(Histogram);
MinHist = min(Histogram);
HoG = (Histogram - MinHist)/(MaxHist-MinHist);
HoG = [1;HoG'];
end