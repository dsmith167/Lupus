# Lupus Time-Series Readme

# Creator: Dylan Smith dsmith167@fordham.edu

# _Contents:_

# **Base Repository (Pre-processing):**

Notes: Current Code requires all csvs to fall in shared folder with the following.

ST1.0\_timesteps.py

	Purpose: Derive statistics about distance between data points

ST1.5\_med\_preprocessing.py

	Purpose: Assign each medication prescription instance a group among pertinent medication groups to Lupus study (or assign to other)

ST2\_transform\_0.1.\_series.py

	Purpose: Transform base data to proper length (3 years of data and 1 year of encoded label data) and synthesize into demographic and encounter derived features

ST3\_lupus\_combine.py

	Purpose: Combine into a cohesive data set, given trimmed and formatted csvs for encounter, lab tests, and medications

ST4A\_time\_series\_impute.py

	Purpose: Given cohesive data set, extrapolate/interpolate forward, and impute the rest of missing data, to output model-ready dataset 1

ST4B\_time\_series\_extrapolate.py

	Purpose: Given cohesive data set, extrapolate in two directions and interpolate, and then remove instances with missing entries, to output model-ready dataset 2



# **Sub Folder (Project Testing):**

Notes: Current Code requires all csvs in outer folder. Additionally, &quot;feat\_importance&quot;, &quot;output&quot;, &quot;ROC curves&quot;, and &quot;saved\_results&quot; are sub folders that need to be established prior to running.

project\_functions.py

	Purpose: Hold functions to be used in project\_test\_dict, for cleanliness and convenience

project\_test\_dict.py

	Purpose: Interactively run a single or the complete set of machine learning models, either from scratch or from previously created data, and export outputs, feature importance, roc auc curves, and saved data
