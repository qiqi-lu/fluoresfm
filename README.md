# Foundation model for fluorescence microscopic image restoration

## DATASET PREPROCESSING
- `0_generate_text.py` : generate the text information used for each dataset.

## MODEL TRAINING :star:

## MODEL EVALUATION

## RESULTS ANALYSIS
After predict the restoraed images using different models, use the following files to analysis the results.
- `4_result_evaluation.py` : calculate the PSNR, SSIM, and ZNCC of the restored images, and saved into a xlsx file. (do for each dataset)
- `4_results_analysis_mean_in.py` : calculate the mean and std of the results in the internal test datasets.
- `4_results_analysis_mean_ex` : calculate the mean and std of the results in the external test datasets.
- `4_results_analysis_pvalue_in.py` : calculate the p-value of the results in the internal test datasets.
- `4_results_analysis_pvalue_ex.py` : calculate the p-value of the results in the external test datasets.

After get all the `mean`, `std`, `n`, and `p-value`, use `4_results_collect_all.py` to collect all the results.