% plot rose diagrams

clear
close all
clc

model_name = '052_Reference_model_10_i';

path_main = pwd;
path_model = [path_main '/' model_name];
path_data = [path_model '/output'];

cd(path_data)
mkdir rose_diagram
path_rose_diagram = [path_data '/rose_diagram'];

list_all = dir('hist_distr_all_*');
list_thr = dir('hist_distr_thresh_*');

n = length(list_all);
bins = 72;

edge_color      = [0, 0, 0,];
line_width      =  0.2;
line_alpha      =  0.2;
face_alpha      =  0.6;
rad_lim         = [0, 1.05];
ang_lim         = [0 360];

for istep = n%1:n
   
    % load files for timestep
    file_all_now = csvread(list_all(istep).name);
    file_thr_now = csvread(list_thr(istep).name);
    
    % get rid of nans in thresholded array
    file_thr_now = rmmissing(file_thr_now);
    
    % mirror data such that distribution is symmetrical over 360 degrees
    % (i.e., strike)
    thr = [file_thr_now (file_thr_now + 180)];
    all = [file_all_now (file_all_now + 180)];
        
    figure(1)
    clf
    set(gcf,'Units','normalized','Position',[.1 .1 .7 .7])
    
    tiledlayout(1,2)
    
    nexttile
    h_all = polarhistogram(deg2rad(all), bins);
    h_all.EdgeColor = edge_color;
    h_all.LineWidth = line_width;
    h_all.EdgeAlpha = line_alpha;
    h_all.FaceAlpha = face_alpha;
    h_all.FaceColor = [0 0.4470 0.7410];
    n_all_samples   = sum(h_all.BinCounts)/2;
    h_all.BinCounts = h_all.BinCounts ./ max(h_all.Values);
    
    hold on
    
    h_all_outline = polarhistogram(deg2rad(all), bins);
    h_all_outline.EdgeAlpha     = 1;
    h_all_outline.LineWidth     = 2;
    h_all_outline.DisplayStyle  = 'stairs';
    h_all_outline.EdgeColor     = 'k'; %[0 0.4470 0.7410];
    h_all_outline.BinCounts     = h_all_outline.BinCounts ./ max(h_all_outline.Values);
    
    
    title(['entire area; N = ' num2str(n_all_samples)])
    rlim(rad_lim)
    thetalim(ang_lim)
    
    hAx=gca;
    hAx.FontSize = 14;
    set(hAx, 'rTick', [0.25, 0.5, 0.75, 1])
    set(hAx, 'rTickLabel', {})
    
    set(gca,'ThetaZeroLocation','top',...
       'ThetaDir','clockwise');
    
    nexttile
    h_thresh = polarhistogram(deg2rad(thr), bins);
    h_thresh.EdgeColor = edge_color;
    h_thresh.LineWidth = line_width;
    h_thresh.EdgeAlpha = line_alpha;
    h_thresh.FaceAlpha = face_alpha;
    h_thresh.FaceColor = [0.8500 0.3250 0.0980];
    n_thresh_samples   = sum(h_thresh.BinCounts)/2;
    h_thresh.BinCounts = h_thresh.BinCounts ./ max(h_thresh.Values);
    
    hold on
    
    h_thresh_outline = polarhistogram(deg2rad(thr), bins);
    h_thresh_outline.EdgeAlpha     = 1;
    h_thresh_outline.LineWidth     = 2;
    h_thresh_outline.DisplayStyle  = 'stairs';
    h_thresh_outline.EdgeColor     = 'k'; %[0.8500 0.3250 0.0980];
    h_thresh_outline.BinCounts     = h_thresh_outline.BinCounts ./ max(h_thresh_outline.Values);

    
    title(['deformed area; N = ' num2str(n_thresh_samples)])
    rlim(rad_lim)
    thetalim(ang_lim)
    
    hAx=gca;
    hAx.FontSize = 14;
    set(hAx, 'rTick', [0.25, 0.5, 0.75, 1])
    set(hAx, 'rTickLabel', {})
    
    set(gca,'ThetaZeroLocation','top',...
        'ThetaDir','clockwise');
    
    sgtitle([num2str(istep * 0.1), ' Ma'],'FontSize', 32) 
    
    
    drawnow
    
%     print('-dpng','-r300','-noui',['rose_diagram/stress_distribution_',...
%         num2str(istep * 0.1,'%2.1f') '_Ma.png'])
    
end

