function [Samples1,Targets1,Samples2,Targets2] = Randomizer(BLMSample,BLMTarget,HoGSample,HoGTarget);

[Row1_1,Column1_1] = size(BLMSample);
[Row2_1,Column2_1] = size(BLMTarget);
Samples1 = zeros(Row1_1,Column1_1); 
Targets1 = zeros(Row2_1,Column2_1);

[Row1_2,Column1_2] = size(HoGSample);
[Row2_2,Column2_2] = size(HoGTarget);
Samples2=zeros(Row1_2,Column1_2); 
Targets2=zeros(Row2_2,Column2_2);

x = randperm(Column1_1);
for i = 1:Column1_1
Randcol = x(i);

Samples1(:,Randcol) = BLMSample(:,i);
Targets1(:,Randcol) = BLMTarget(:,i);

Samples2(:,Randcol) = HoGSample(:,i);
Targets2(:,Randcol) = HoGTarget(:,i);
end

end