% make input structure from files
%names = {'booksh', 'box', 'castle', 'corr', 'graff', 'head', 'kampa', 'Kyoto', 'leafs', ...
%	'plant', 'rotunda', 'shout', 'valbonne', 'wall', 'wash', 'zoom'};
%exts = {'png', 'png', 'png', 'png', 'png', 'jpg', 'png', 'jpg', 'jpg', 'png',...
%	'png', 'png', 'png', 'jpg', 'png', 'png'};

names = {'Brussels', 'Dresden', 'Leuven1', 'Leuven2', 'dino1', 'dino2', 'temple1', 'temple2'};
exts = {'jpg', 'jpg', 'jpg', 'jpg', 'jpg', 'jpg', 'jpg', 'jpg'};

path = [pwd '/'];

load('input');

for i = 1 : size(names, 2)
	names{i}
    
    load([names{i} '_vpts']);
        
    final_pts = [[input(i).u; zeros(1, size(input(i).u, 2))] [validation.pts; ones(1, size(validation.pts, 2))]];
       
    dlmwrite([names{i} '_pts.txt'], final_pts', 'delimiter', ' ', 'precision', 100)   
    %dlmwrite([names{i} '_model.txt'], validation.model, 'delimiter', ' ', 'precision', 100)   
    
    pts1 = [input(i).u(1:2,:) validation.pts(1:2,:)]';
    pts2 = [input(i).u(4:5,:) validation.pts(4:5,:)]';
    
    continue;
    
    stats = zeros(2, 4);
    for rep = 1 : 1000
        tic;
        [F, inliersIndex, status] = estimateFundamentalMatrix(pts1, pts2, 'Method','RANSAC', 'NumTrials', 10000, 'Confidence', 0.95, 'DistanceThreshold', 2.00, 'DistanceType', 'Sampson');
        elapsedTime = toc;
        
        distance = 0;
        for j = 1 : size(validation.pts, 2)
            pts1h = validation.pts(1:3,j);
            pts2h = validation.pts(4:6,j);

            a = pts2h' * F;
            b = F * pts1h;
            c = pts2h' * b;
            
            d = a * a' + b' * b;

            distance = distance + abs(c(1) / d);
        end
        ransac_distance = distance / size(validation.pts, 2);
        
        stats(1,1) = stats(1,1) + elapsedTime / 1000;
        stats(1,2) = stats(1,2) + ransac_distance / 1000;
        stats(1,3) = stats(1,3) + elapsedTime / 1000;
        stats(1,4) = stats(1,4) + elapsedTime / 1000;
    end
    stats
    
    return;
    
    tic
    [F, inliersIndex] = estimateFundamentalMatrix(pts1, pts2, 'Method','MSAC', 'NumTrials', 10000, 'Confidence', 0.95, 'DistanceThreshold', 0.31, 'DistanceType', 'Sampson');
    elapsedTime = toc
    
    distance = 0;
    for j = 1 : size(validation.pts, 2)
        pts1h = validation.pts(1:3,j);
        pts2h = validation.pts(4:6,j);
        
        pfp = (pts2h' * F)';
        pfp = pfp .* pts1h;
        d = sum(pfp, 1) .^ 2;
        
        epl1 = F * pts1h;
        epl2 = F' * pts2h;
        d = d ./ (epl1(1,:).^2 + epl1(2,:).^2 + epl2(1,:).^2 + epl2(2,:).^2);
        
        distance = distance + d;
    end
    msac_distance = distance / size(validation.pts, 2)
        
    
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
