
# Residential Property Transaction Predictive Model
This code is used to build a machine learning model used to predict the price per sqft of resale properties in Singapore, with input of the year, property type, and postal district.

The files below utilizes Tensorflow, Pandas, Keras, scikit-learn to be trained on property transactions in Singapore from May 2018 to May 2023.
The dataset it utilizes contains the transactions' Project Name, Transacted Price, Area (SQFT), Unit Price ($ PSF), Sale Date, Property Type, Postal District.
Dataset is obtained from https://www.ura.gov.sg/property-market-information/pmiResidentialTransactionSearch

Relevant Files:
- resale-trend-ipynb.ipynb
- resale-trend.py
- ResidentialTransaction20230510192045.xlsx


The files below trains a model to read CPI trends in Singapore to predict inflation rate and CPI in given year and quarter. It utilizes much of the same libraries to build the model, with only a few additions.
The dataset it utilizes contains the year, quarter, inflation rate and the CPI, for the sake of simplicity, CPI from 1961 2Q is set at 100.
Dataset is obtained from https://tablebuilder.singstat.gov.sg/table/TS/M213161

Relevant Files:
- CPI-trend-Singapore.ipynb
- CPI-trend-Singapore.py
- M213161.xlsx

## Roadmap
- Improve model accuracy for wider range of years.
- Compare inflation rates over time with the trend of resale property prices.
- Compare inflation rates with unemployment rate in Singapore.


## Notes

This model was created as I wanted to develop a site to showcase the comparison between rising resale property prices juxtaposed with inflation in Singapore. In addition, I wanted to look at the relation between inflation and unemployment in Singapore, and compare it with the old adage, "If the unemployment rate goes up 1%, 40,000 more people die", scale it down for it to be applicable to Singapore, and determine whether it remains as accurate.
I believe large purchases such as properties can help us determine more accurately how different parts of the market can create domino effects rapidly and quickly become out of control, and as such I started with this model to work my way forward.
