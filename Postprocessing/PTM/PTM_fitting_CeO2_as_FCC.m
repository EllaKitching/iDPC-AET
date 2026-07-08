%% calculate PDF, mean bond length and Nearest Neighbor threshold (first-nearest-neighbor shell distance)

clear
clc 
%%
% load atom positions
load("ceonly_4366_d4_wienerfilt")
res = 1;% 1if in Angstrom, if not px 
%%
coords = atom_pos.';         % transpose if coords is 3×N → now N×3
N_atoms = size(coords, 1); % number of atoms (rows)
types = repmat({'Ce'}, N_atoms, 1);  % one type per row

%%
coords = ce_coords; 
%%
isCe = strcmp(types, 'Ce');
coords = coords(isCe, :);% change to be columns or rows accordingly
disp(size(coords, 1));

%%
pos_unique = unique(coords, 'rows').';  % unique columns in [3 x N] format vs row
minAllowedDist = 1;  
pos = filterCoordsByMinDist(pos_unique, minAllowedDist);
%%
doPlot = 0;
[LatticeConstOG, MeanBondLengthOG, ThresholdOG, PDF] = CalculateLatticeConstantDebugAsym(pos,res,'fcc',doPlot);

dm = squareform(pdist(pos'));
min_nonzero = min(dm(dm > 0));
fprintf('Min nonzero distance: %.6f Å\n', min_nonzero);
max_dist = max(dm(:));
fprintf('Max distance in dataset: %.3f Å\n', max_dist); % check to apply suitable histgram bisn and range - fails otherwise

% check if parallel ready
if canUseParallelPool
    disp("Parallel Computing Toolbox is installed")
else
    disp("Parallel Computing Toolbox is not installed")
end

%% polyhedral template matching
% load atom position and type first, use PB as example

doPlot = 1;

dm = squareform(pdist(pos));
min_nonzero = min(dm(dm > 0));
fprintf('Min nonzero distance: %.6f Å\n', min_nonzero);
max_dist = max(dm(:));
fprintf('Max distance in dataset: %.3f Å\n', max_dist); % check to apply suitable histgram bisn and range - fails otherwise

[LatticeConst, MeanBondLength, Threshold, PDF] = CalculateLatticeConstantAsym(pos,res,'fcc',doPlot);

%% polyhedral template matching
fprintf('PTM calcs started! Current time %s\n', datestr(now,'HH:MM:SS'))
PTMResult = PTM01Asym(pos, res);
fprintf('PTM01 done! Current time %s\n', datestr(now,'HH:MM:SS'))
%%
PTMResult = PTM02(PTMResult, 1, 1, 5, 50);
% PTM02(PTMResult, AdaptiveFitMode, DrawFlag, numIterICP, numTheta, scoreThreshold)
fprintf('PTM02 done! Current time %s\n', datestr(now,'HH:MM:SS'))
%%
% draw results clearly
figure;
subplot(131)
plotAtoms(PTMResult.xyz(PTMResult.ind_fcc,:)','fcc',{'#D95319',10})
plotAtoms(PTMResult.xyz(PTMResult.ind_hcp,:)','hcp',{'#0072BD',30})
subplot(132)
plotAtoms(PTMResult.xyz(PTMResult.ind_hcp,:)','hcp',{'#0072BD',30})
subplot(133)
plotAtoms(PTMResult.xyz(PTMResult.ind_chaos,:)','others',10)

%% save figs - new add
figHandles = findall(0, 'Type', 'figure');

% Loop through each figure and save as PNG
for i = 1:length(figHandles)
    figure(figHandles(i)); % Make it the current figure
    filename = sprintf('wienerAsym/figure/figure_%d_ceonly_4366_d4_wienerfilt.png', i);
    saveas(figHandles(i), filename); % or exportgraphics for better quality
end

%% notes on PTM outputs
% PTM output data (e.g., fccData, hcpData, dhData, ihData) is stored as [N x 7] arrays:
%   Col 1: RMSD of fitted template (Å)
%   Col 2: Lattice constant fitted for this atom (Å)
%   Col 3: Maximum bond length used in template fitting (Å)
%   Col 4-7: Quaternion components [qw, qx, qy, qz] representing orientation
%             (qw is scalar part, following OVITO convention)
%What each column means, For each atom:
%Column 1 – Best similarity score (before ICP) for FCC among all tested orientations.
%(Comes from tempData(2,1) = max(score_FCC(:)))
%Column 2 – Orientation index (a0) of the best FCC orientation.
%(Which row of uvwOrient gave the best score.)
%Column 3 – Rotation index (a1) within the phiArray that gave the best score.
%(Basically which in-plane rotation was optimal.)
%Column 4 – uvwOrient_x, the x-component of the chosen FCC orientation vector.
%(Filled from uvwOrient(Column 2,:))
%Column 5 – uvwOrient_y, the y-component of the chosen FCC orientation vector.
%Column 6 – uvwOrient_z, the z-component of the chosen FCC orientation vector.
%Column 7 – Best FCC similarity score after ICP refinement.
%(Comes from tempData(2,4))

save("wienerAsym\ceonly_4366_d4_wienerfilt_AsymPTMResult50.mat", "PTMResult")
save("wienerAsym\ceonly_4366_d4_wienerfilt_Asym_LatticeConst50.mat", "LatticeConst")

%%
function filtered_coords = filterCoordsByMinDist(coords, minDist)
% coords: 3xN matrix of positions
% minDist: minimum allowed distance between any two points (e.g., 0.5 Å)

N = size(coords, 2);
keep = true(1, N);

for i = 1:N
    if ~keep(i)
        continue;
    end
    % compute distances from coords(:,i) to all coords(:,j > i)
    dists = sqrt(sum((coords(:, i+1:end) - coords(:, i)).^2, 1));
    tooCloseIdx = find(dists < minDist) + i;
    keep(tooCloseIdx) = false;  % remove those points
end

filtered_coords = coords(:, keep);

fprintf('Filtered coords from %d to %d by minDist=%g Å\n', N, sum(keep), minDist);


end

