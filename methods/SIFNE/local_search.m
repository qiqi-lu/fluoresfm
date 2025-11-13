% Copyright (c) 2016.
% All rights reserved. Please read the 'license.txt' for license terms.
% 
% Developers: Zhen Zhang, Pakorn Kanchanawong
% Contact: biekp@nus.edu.sg
function index_global_pool  = local_search_display(L_GlobalIndex, label_list,  fanAngle,  all_tips, r,   base_ori,  base_loc,   Min_Info, C1weight, C3weight)
%  global index in label list/list of all tips
MIN_Angle_Diff = Min_Info(1);
MIN_Dist = Min_Info(2);
MIN_GapAngle_Diff = Min_Info(3);

% load data\AllFragments.mat;
% load data\OriginImg.mat;
% load data\R.mat;

All_Dist = sqrt( (all_tips(:,1)-base_loc(1)).^2  +  (all_tips(:,2)-base_loc(2)).^2);
All_Dist(find(All_Dist==0)) = 3*MIN_Dist;

index_global_pool = find(All_Dist<=MIN_Dist);

% check 1: calculate their angle difference with base orientation
if ~isempty(index_global_pool)
    for i = 1:length(index_global_pool)
        CanAngle = check_dir_between2pts(base_loc,all_tips(index_global_pool(i),1:2));
        Diff = abs(CanAngle - base_ori);
        if Diff>180
            Diff = 360 - Diff;
        end
        if Diff>(fanAngle/2)
            index_global_pool(i) = 0;
        end
    end
    index_global_pool(find(index_global_pool==0)) = []; % keep all points in the sector region (fan region)
    
    if ~isempty(index_global_pool) % check 2
        CandidateTips = all_tips(index_global_pool,:);
        CandidateTips(:,7) = index_global_pool;
        
        
        % ****** structure of all_tips so far ******   #: Number
        %           colume1            colume2           colume3         colume4         colume5            colume6
        % tip1     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center
        % tip2     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center
        % tip3     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center
        % tip4     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center
        % ......
        % ****** structure of all_tips so far ******   #: Number
        
        % ****** structure of CandidateTips so far ******   #: Number
        %                    colume1            colume2           colume3         colume4         colume5            colume6        colume7
        % CandidateTip1     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center    global_index
        % CandidateTip2     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center    global_index
        % CandidateTip3     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center    global_index
        % CandidateTip4     # of Row           # of Col           Labeled #     Orientation     Row # Center      Col # Center    global_index
        % ......
        % ****** structure of CandidateTips so far ******   #: Number
        
        % ****** check condition 1 ******
        
        % calculate orientation differences below
        for i = 1:size(CandidateTips,1)
            if CandidateTips(i,4)==0
                CandidateDir(i) = -180;
            else
                CandidateDir(i) = CandidateTips(i,4) - sign(CandidateTips(i,4))*180;
            end
        end
        angle_diff_list = abs(CandidateDir - base_ori);
        angle_diff_list(find(angle_diff_list>180)) = 360 - angle_diff_list(find(angle_diff_list>180));
        % calculate orientation differences above
        % ****** check condition 1 ******
        
        
        
        % ****** check condition 3 ******
        
        % check gap difference below
        MIN3_List = [];
        for i = 1:size(CandidateTips,1)
            if angle_diff_list(i)<MIN_GapAngle_Diff
                MIN3_List = [MIN3_List,  mean([angle_diff_list(i), MIN_GapAngle_Diff])];
            else
                MIN3_List = [MIN3_List,  MIN_GapAngle_Diff];
            end
            GapOrient(i) = check_dir_between2pts(base_loc,CandidateTips(i,1:2));
        end
        GapOrient_Diff = abs(GapOrient - base_ori);
        GapOrient_Diff(find(GapOrient_Diff>180)) = 360 - GapOrient_Diff(find(GapOrient_Diff>180));
        % check gap difference above
        % ****** check condition 3 ******
        
        ShortListCandidateTips = [];
        for i = 1:size(CandidateTips,1)
            Min1 = angle_diff_list(i)<=MIN_Angle_Diff;
            Min3 = GapOrient_Diff(i)<=MIN3_List(i);
            
            if  Min1 && Min3
                ShortListCandidateTips = [ShortListCandidateTips;CandidateTips(i,:)];
            else
                index_global_pool(i) = 0;
            end
        end
        angle_diff_list(find(index_global_pool==0)) = [];
        GapOrient_Diff(find(index_global_pool==0)) = [];
        index_global_pool(find(index_global_pool==0)) = [];
        
        if ~isempty(ShortListCandidateTips)
            % remove tips whose inner fragments exist; another concern is that an entire fragment is included in the searching region. this is ok because farther tip could be removed according to condition1
            delete_list = [];
            FanTipsX = CandidateTips(:,1);
            FanTipsY = CandidateTips(:,2);
            
            for k = 1:size(ShortListCandidateTips,1)
                temp_label_list = CandidateTips(:,3); % with information of tips which may have been removed
                temp_label_list(find(temp_label_list==ShortListCandidateTips(k,3)))=[];
                uniq_label = unique(temp_label_list);
                for j = 1:length(uniq_label)
                    if length(find(temp_label_list==uniq_label(j)))==1
                        continue;
                    else
                        pair_indice = find(CandidateTips(:,3)==uniq_label(j));
                        if pdist([base_loc;ShortListCandidateTips(k,1:2)]) > pdist([base_loc;CandidateTips(pair_indice(1),1:2)])...
                                && pdist([base_loc;ShortListCandidateTips(k,1:2)]) > pdist([base_loc;CandidateTips(pair_indice(2),1:2)])
                            inTip_idx1 = dsearchn([FanTipsX(pair_indice(1))  FanTipsY(pair_indice(1)); FanTipsX(pair_indice(2))  FanTipsY(pair_indice(2))],base_loc);
                            inTip1 = [FanTipsX(pair_indice(inTip_idx1))  FanTipsY(pair_indice(inTip_idx1))];
                            inTipAngleTemp1 = all_tips(L_GlobalIndex(sub2ind(size(L_GlobalIndex), inTip1(1), inTip1(2))),4);
                            % C1-close
                            if inTipAngleTemp1==0
                                inTipAngle1 = -180;
                            else
                                inTipAngle1 = inTipAngleTemp1 - sign(inTipAngleTemp1)*180;
                            end
                            inAngleDiff1 = abs(inTipAngle1 - base_ori);
                            if inAngleDiff1>180
                                inAngleDiff1 = 360-inAngleDiff1;
                            end
                            % C1-close
                            
                            %C3-close
                            inGapAngle1 = check_dir_between2pts(base_loc,inTip1);
                            inGapOrientDiff1 = abs(inGapAngle1 - base_ori);
                            if inGapOrientDiff1>180
                                inGapOrientDiff1 = 360-inGapOrientDiff1;
                            end
                            %C3-close
                            
                            inTip_idx2 = dsearchn([FanTipsX(pair_indice(1))  FanTipsY(pair_indice(1)); FanTipsX(pair_indice(2))  FanTipsY(pair_indice(2))],ShortListCandidateTips(k,1:2));
                            inTip2 = [FanTipsX(pair_indice(inTip_idx2))  FanTipsY(pair_indice(inTip_idx2))];
                            inTipAngleTemp2 = all_tips(L_GlobalIndex(sub2ind(size(L_GlobalIndex), inTip2(1), inTip2(2))),4);
                            % C1-far
                            if inTipAngleTemp2==0
                                inTipAngle2 = -180;
                            else
                                inTipAngle2 = inTipAngleTemp2 - sign(inTipAngleTemp2)*180;
                            end
                            inAngleDiff2 = abs(inTipAngle2 - all_tips(L_GlobalIndex(sub2ind(size(L_GlobalIndex), ShortListCandidateTips(k,1),ShortListCandidateTips(k,2))),4));
                            if inAngleDiff2>180
                                inAngleDiff2 = 360-inAngleDiff2;
                            end
                            % C1-far
                            
                            %C3-far
                            inGapAngle2 = check_dir_between2pts([ShortListCandidateTips(k,1),ShortListCandidateTips(k,2)],inTip2);
                            inGapOrientDiff2 = abs(inGapAngle2 - all_tips(L_GlobalIndex(sub2ind(size(L_GlobalIndex), ShortListCandidateTips(k,1),ShortListCandidateTips(k,2))),4));
                            if inGapOrientDiff2>180
                                inGapOrientDiff2 = 360-inGapOrientDiff2;
                            end
                            %C3-far
                            
                            
                            % now check whether the inner filament is really an intermediate  bridge
                            if inAngleDiff1<=MIN_Angle_Diff && inGapOrientDiff1<=MIN_GapAngle_Diff && inAngleDiff2<=MIN_Angle_Diff && inGapOrientDiff2<=MIN_GapAngle_Diff
                                delete_list = [delete_list,k];
                                continue;
                            end
                        end
                    end
                end
            end
            angle_diff_list(delete_list) = [];
            GapOrient_Diff(delete_list) = [];
            index_global_pool(delete_list) = [];
            ShortListCandidateTips(delete_list,:) = [];
            
            
            % sort index_global_pool based on scoring system below
            if ~isempty(index_global_pool)
                if length(index_global_pool)>=2
                    C1_score = angle_diff_list/max(angle_diff_list);
                    C3_score = GapOrient_Diff/max(GapOrient_Diff);
                    C = C1_score*C1weight + C3_score*C3weight;
                    [B,I] = sort(C);
                    index_global_pool = index_global_pool(I);
                    index_global_pool = [index_global_pool',   zeros(1,(100-length(index_global_pool)))];
                else
                    index_global_pool = [index_global_pool',   zeros(1,99)];
                end
            else
                index_global_pool = zeros(1,100);
            end
            
            % sort index_global_pool based on scoring system above
            
            % remove tips whose inner fragments exist; another concern is that an entire fragment is included in the searching region. this is ok because farther tip could be removed according to condition1
            
            
            % ****** structure of RecordInfo so far ******   #: Number
            % Global_Index	x	y	Label	Orienttaion	               *                              *					             *				                  	*           	           	             	*
            %
            % Candidate1    x	y	Label	Orienttaion	condition1(Angle_Diff)Accept?	condition2(Dist_Diff)Accept?	condition3(GapOrient_Diff)Accept?	  condition4(GeoError)Accept?            condition5(LengthRatio)Accept?
            %      *        *   *     *         *                  Angle_Diff                       Dist_Diff                      GapOrient_Diff                     ErrorIndex                               LengthRatio
            %
            % Candidate2    x	y	Label	Orienttaion	condition1(Angle_Diff)Accept?	condition2(Dist_Diff)Accept?	condition3(GapOrient_Diff)Accept?	  condition4(GeoError)Accept?            condition5(LengthRatio)Accept?
            %      *        *   *     *         *                  Angle_Diff                       Dist_Diff                      GapOrient_Diff                     ErrorIndex                               LengthRatio
            %
            % Candidate3    x	y	Label	Orienttaion	condition1(Angle_Diff)Accept?	condition2(Dist_Diff)Accept?	condition3(GapOrient_Diff)Accept?	  condition4(GeoError)Accept?            condition5(LengthRatio)Accept?
            %      *        *   *     *         *                  Angle_Diff                       Dist_Diff                      GapOrient_Diff                     ErrorIndex                               LengthRatio
            %
            % Candidate4   	x	y	Label	Orienttaion	condition1(Angle_Diff)Accept?	condition2(Dist_Diff)Accept?	condition3(GapOrient_Diff)Accept?	  condition4(GeoError)Accept?            condition5(LengthRatio)Accept?
            %      *        *   *     *         *                  Angle_Diff                       Dist_Diff                      GapOrient_Diff                     ErrorIndex                               LengthRatio
            %
            % ......
            % ......
            % ****** structure of RecordInfo so far ******   #: Number
            
        else
            index_global_pool = zeros(1,100);
        end
        
    else
        index_global_pool = zeros(1,100);
    end
else
    index_global_pool = zeros(1,100);
end

