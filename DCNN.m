function dcnnf = DCNN(Training)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DCNN特徴を抽出
% net準備
net = alexnet;

%%%%%%%%%%%%%%%% 特徴抽出
for j=1:size(Training,1)
    % 入力画像を準備
    img = imread(Training{j});
    reimg = imresize(img,net.Layers(1).InputSize(1:2));
    if(j == 1)
        IM = reimg;
    else
        IM = cat(4,IM,reimg);
    end
end


% activationsを利用して中間特徴量を取り出します．
% 4096次元の'fc7'から特徴抽出します．
dcnnf = activations(net,IM,'fc7');  

% squeeze関数で，ベクトル化します．
dcnnf = squeeze(dcnnf);

% L2ノルムで割って，L2正規化．
% 最終的な dcnnf を画像特徴量として利用します．
dcnnf = dcnnf/norm(dcnnf);


% このままでは逆なので転置
dcnnf = dcnnf';