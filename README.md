# Engine Scan Analyzer

This script was created to analyze the results of CT Scans done on Space Concordia's Composite Engines.

## Basic Concepts

At the bottom of this page the reader may find some useful resources that were found through research. These resources contain additional information to what will be provided in this section.

The most basic concept necessary to understand the procedure is how Hounds Field Units work. Essentially, the proclivity for a material to reflect an energy beam depends on this unit.

The Hounds Field Unit utilized in this script at the base is [-400, 200]. This may be incorrect and it is highly encouraged to try other ranges.

**Hounds Field Unit**: Hounsfield units (HU) are a dimensionless unit universally used in computed tomography (CT) scanning to express CT numbers in a standardized and convenient form. Hounsfield units are obtained from a linear transformation of the measured attenuation coefficients.

**Attenuation coefficient**: The attenuation coefficient is a measure of how easily a material can be penetrated by an incident energy beam.

**Level**: the level in the script is the center of the range for the Hounds Field Unit.

**Window**: the window defines the upper and lower boundary of the Hounds Field Unit range.

**For example**: The level being used in the script is -100 and the window is -300. This creates the range [-400, 200].

## Getting Started

Install the packages used throughout the project. It is highly recommended to use a virtual environment.

The flow is the following:
1- Process and save the CT scans locally
2- Load the CT Scans and calculate the average density

## Improvements

The proper level and window for the script to work are to be determined. 

Sometimes the contour finding doesn't work properly. The current quick fix for this is to avoid counting any densities under 80%.

## Resources
https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
https://www.kaggle.com/code/allunia/pulmonary-dicom-preprocessing
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5462462/
https://www.youtube.com/watch?time_continue=194&v=KZld-5W99cI&feature=emb_logo&ab_channel=PetraLewis
https://radiopaedia.org/articles/hounsfield-unit#:~:text=Hounsfield%20units%20(HU)%20are%20a,the%20measured%20attenuation%20coefficients%201.
