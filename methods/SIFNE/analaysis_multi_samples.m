% maltiple sample analysis
id_smaple_list = [41,42,43,44,45,46,47,48];
for id_sample = id_smaple_list
    imgpath = "E:\qiqilu\Project\2024 Foundation model\code\results\predictions\biosr-mt-sr-2-in-ccp\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16\"+num2str(id_sample)+".tif";
    func_analysis_mt(imgpath)
    % imgpath = "E:\qiqilu\datasets\BioSR\transformed\MTs\test\channel_0\SIM\"+num2str(id_sample)+".tif";
    % func_analysis_mt(imgpath)
    % imgpath = "E:\qiqilu\datasets\BioSR\transformed\MTs\test\channel_0\WF_noise_level_2_up2\"+num2str(id_sample)+".tif";
    % func_analysis_mt(imgpath)
    imgpath = "E:\qiqilu\Project\2024 Foundation model\code\results\predictions\biosr-mt-sr-2-in-er\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16\"+num2str(id_sample)+".tif";
    func_analysis_mt(imgpath)
end
