%% Classifier combination (BlockMean && Histogram Of Gradient))

%% Read Image And Compute Means Of Block Images
N = 400; % Number Of Images*
m = 40; % Digits 0~9*
W = input('With Of Window: ');
H = input('Length Of Window: ');
WinSize = input('Length Of Squre For Avaraging: ');

BLMSample = zeros(((H/WinSize)*(W/WinSize))+1,N);
BLMTarget = zeros(m,N);
HoGSample = zeros(182,N);
HoGTarget = zeros(m,N);


n = 1;
for a = 1 : 40
    for b = 1 : 10
        Adress = ['ORL\s',num2str(a),'\',num2str(b),'.pgm'];
        if(exist(Adress,'file')) ~= 0
            I = imread(Adress);
            
            % Block Means Features
            FeatureVec1 = blockMeanFeatures(I,H,W,WinSize); % Get Features From FeatureExtraction Function*
            BLMSample(:,n) = FeatureVec1; % Matrix Consist Of Block Means Features* 
            BLMTarget(a,n) = 1; % Make Target Matrix That Evry Column Has One True Value*
            
            % HoG Features
            FeatureVec2 = HoGFeatures(I,H,W);
            HoGSample(:,n) = FeatureVec2; % Matrix Consist Of Histogram Of Gradian Features*
            HoGTarget(a,n) = 1; % Target Matrix Related To HoG Features*
            
            n = n+1;
        end
    end
end

% Randomize Samples For Learning Network
[Samples1,Targets1,Samples2,Targets2] = Randomizer(BLMSample,BLMTarget,HoGSample,HoGTarget);


TrainCont = round(.7*N);
TrainSample1 = Samples1(:, 1:TrainCont); % Data For Train = %70 Samples*
TrainTarget1 = Targets1(:, 1:TrainCont); % Target For Train Data*
TestSample1  = Samples1(:, TrainCont:N); % Data For Test = %30 Samples*
TestTarget1  = Targets1(:, TrainCont:N); % Target For Test Data*
TrainSample2 = Samples2(:, 1:TrainCont); % Data For Train = %70 Samples*
TrainTarget2 = Targets2(:, 1:TrainCont); % Target For Train Data*
TestSample2  = Samples2(:, TrainCont:N); % Data For Test = %30 Samples*
TestTarget2  = Targets2(:, TrainCont:N); % Target For Test Data*

%% Training Section
Alpha = input('Learning Rate: ');
NH1 = input('Number Of Neuron For Hidden Layer (BlocKMean Features): ');
NH2 = input('Number Of Neuron For Hidden Layer (HoG Features): ');
Epoch = input('Number of Epoch: ');
[Wih1,Who1] = classiferBlockMean(TrainSample1,TrainTarget1,TrainCont,Alpha,NH1,Epoch,H,W,WinSize,m);
[Wih2,Who2] = classifierHoG(TrainSample2,TrainTarget2,TrainCont,Alpha,NH2,Epoch,m);

%% Test Section

Cont = 0;
for i = 1 : N - TrainCont
    
    OSample1 = TestSample1(:,i);
    OTarget1 = TestTarget1(:,i);
    OSample2 = TestSample2(:,i);
    OTarget2 = TestTarget2(:,i);
    
    % OutPut For Block Mean Features
    NETi = OSample1'*Wih1;
    NETi = NETi';
    OUTi = 1./(1+exp(-NETi));
    Xi = [1;OUTi];
    NET1 = Xi'*Who1;
    NET1 = NET1';
    OUT1 = 1./(1+exp(-NET1));
    
    % OutPut For HoG Features
    NETj = OSample2'*Wih2;
    NETj = NETj';
    OUTj = 1./(1+exp(-NETj));
    Xj = [1;OUTj];
    NET2 = Xj'*Who2;
    NET2 = NET2';
    OUT2 = 1./(1+exp(-NET2));
    
    % Avrage Of OutPuts
    OUT = (OUT1 + OUT2)./2;

    [V, Index] = max(OUT);
    if(OTarget1(Index) == 1)
        Cont = Cont + 1;
    end
    
end

SehateDorosti = (Cont/(N-TrainCont))*100