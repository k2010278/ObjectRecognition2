function result = main2(num)

%%ポジティブ画像の読み込み
list3=textread('urllist3.txt','%s');
OUTDIR3='imgdir3';
if ~(exist('imgdir3/','dir') == 2)
    mkdir(OUTDIR3);
    for i=1:size(list3,1)
        fname3=strcat(OUTDIR3,'/',num2str(i,'%04d'),'.jpg')
        websave(fname3,list3{i});
    end
end

%ネガティブ画像の読み込み
n = 0; list4 = {};
LIST={'bgimg'};
DIR0='bgimage/';
     
for i = 1:length(LIST)
    DIR = strcat(DIR0,LIST(i),'/');
    W = dir(DIR{:});

    for j = 1:size(W)
        if (strfind(W(j).name,'.jpg'))
            fn = strcat(DIR{:},W(j).name);
            n=n+1;
        
            list4 = [list4(:)' {fn}];
        end
    end
 end

%ノイズが多い画像の読み込み
list5=textread('urllist5.txt','%s');
OUTDIR5='imgdir5';
% imgdirが存在しないならフォルダを作成して画像を呼び込む
if ~(exist('imgdir5/','dir') == 2)
    mkdir(OUTDIR5);
    for i=1:size(list5,1)
        fname5=strcat(OUTDIR5,'/',num2str(i,'%04d'),'.jpg')
        websave(fname5,list5{i});
    end
end


% ポジティブ、ネガティブの画像の設定
PosList=list3(1:num);   
NegList=list4';
Training=[PosList; NegList];

%ノイズの多い画像の設定
Training2=list5;

% 学習モデル用のAlexNetによるDCNN特徴量計算
dcnnf = DCNN(Training);
% ノイズが多い画像セット300枚のDCNN特徴抽出
dcnnf2 = DCNN(Training2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%線形SVM

%labelは(num+500)*1行列
training_label = [ones(num,1); ones(500,1)*(-1)];

%training_dataはポジティブ+ネガティブ
training_data = dcnnf;

%test_dataはノイズの多い画像から得たdcnnf
test_data = dcnnf2;

%学習
model = fitcsvm(training_data, training_label,'KernelFunction','linear'); 

%分類
[predicted_label, score] = predict(model, test_data);


% 降順 ('descent') でソートして，ソートした値とソートインデックスを取得
[Score, Idx] = sort(score(:,2), 'descend');

% list{:} に画像ファイル名が入っているとして，
% sorted_idxを使って画像ファイル名，さらに
% sorted_score[i](=score[sorted_idx[i],2])の値を出力します．
for i=1:numel(Idx)
  fprintf("%s %f\n", list5{Idx(i)}, Score(i));
end



%画像の表示（再ランキング）
for i=1:100
    subplot(10,10,i), imshow(imread(list5{Idx(i)}));
end

%画像の表示(再ランキングなし)
for i=1:100
    subplot(10,10,i), imshow(imread(list5{i}));
end

result = "finish";