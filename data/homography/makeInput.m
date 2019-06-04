% make input structure from files
names = {'adam' 'boat' 'Boston' 'BostonLib' 'BruggeSquare' 'BruggeTower' 'Brussels' 'CapitalRegion' 'city' 'Eiffel' 'ExtremeZoom' 'graf' 'LePoint1' 'LePoint2' 'LePoint3' 'WhiteBoard'};
exts = {'png' 'png' 'png' 'png' 'jpg' 'png' 'jpg' 'jpg' 'png' 'png' 'png' 'png' 'jpg' 'png' 'png'};
path = [pwd '/'];

load('homogr');

for i = 1 : size(names, 2)
	names{i}
    
    load([names{i} '_vpts']);
        
    final_pts = [[cell2mat(results.u(i)); zeros(1, size(cell2mat(results.u(i)), 2))] [validation.pts; ones(1, size(validation.pts, 2))]];
    %final_pts = [validation.pts; ones(1, size(validation.pts, 2))];
        
    dlmwrite([names{i} '_pts.txt'], final_pts', 'delimiter', ' ', 'precision', 100)   
        
    %dlmwrite([names{i} '_model.txt'], validation.model, 'delimiter', ' ', 'precision', 100)   
    
% 	set(GUI.main.h.limage, 'String', [names{i} 'A.' exts{i}]);
% 	delete(findobj(allchild(GUI.figh1),'flat','serializable','on'));
% 	load_img(1, [path names{i}, 'A.', exts{i}]); zoom(GUI.figh1,'out');
% 
% 	set(GUI.main.h.rimage, 'String', [names{i}, 'B.', exts{i}]);
% 	delete(findobj(allchild(GUI.figh2),'flat','serializable','on'));
% 	load_img(2, [path names{i}, 'B.', exts{i}]); zoom(GUI.figh2,'out');
% 
% 	RES = [];
% 	do_all('eg');
% 	
% 	load([names{i} '_vpts']);
% 	
% 	input(i).name = names{i};
% 	input(i).ext = exts{i};
% 	input(i).u = TC.u(1:6,:);
% 	input(i).th = RES.model.params.th;
% 	input(i).validation = validation;
	
end

%save inputHA input
