# Analyses for the paper "Global prevalence of setting longlines at dawn highlights bycatch risk for threatened albatross".

This folder contains notebooks that include the analyses and figures for the paper.


**Brazil_analysis-v20220718.py**: This notebook investigates how sets predicted by the model match with sets recorded in logbooks.

**CompareRFMOhooks2GFWSets-GlobalPlots.py**: This notebook produces the ratio of hooks to sets mentioned in the Results section of the paper.

**Fig1andAreaAnalysis.py**: This notebook produces Figure 1 from the paper, and provides an estimate for how much of the ocean is within 30km of a set in a given time frame.

**Fig2andSetsAnalysis.py**: This notebook produces Figure 2 from the paper, number of vessels, number of sets, and set durations.

**Fig3.py**: This notebook produces Figure 3.

**Fig4.py**: This notebook produces Figure 4, which shows the extent to which albatross ranges overlap with longline sets.

**Make_longline_tables-v20220802.py**: This notebook produces the bigquery tables for the night ratio analysis.

**longline_groundtruth_sets_v20220802.csv**: This CSV contains the ground truth test set labels used in the model evaluation.

**longline_test_days_v20220613.csv**: This csv contains the pseudo-randomly selected day and ssvid combinations used for the model evaluation.

**longline_test_evaluation_v20220802.py**: This notebook produces the precision and recall for the model.

**night_ratio_analysis.py**: This notebook produces the night setting ratios for Table 1.

