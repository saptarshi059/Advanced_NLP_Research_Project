# Advanced Natural Language Processing Final Project

**Code and Project Advisor: Dr. Ted Pedersen</br>
Code Author: Saptarshi Sengupta (sengu059@d.umn.edu)</br>
Dept. of Computer Science, University of Minnesota Duluth**

### Introduction ###

This repository contains the code for my project on the Author Profiling task at PAN 2013, for my Advanced NLP course. The task was simple; given a document, determine the authors age and gender. These models were not officially deployed for the task. They were developed to see how deep learning architectures would fair on this task. The following headings explain a bit more about the contents of each folder.

### Data Details ###

Unfortunately, the official data could not be uploaded as GitHub does not allow file sizes above 100MB. However, one can download the official data from https://zenodo.org/record/3715864#.XrYEphZOmFA, by requesting permission. I provided sample data which I used to test my programs on. These are snippets of the actual data.

### Notebooks ###

I originally developed these programs with Jupyter. However, for simplicity, I also provide .py versions of the notebooks which runs through all the code at once. These notebooks also contain some extra code which I was fiddling around with which might come in handy!

### My Paper ###

I tabulated my findings in a self-authored paper which can be found in the "My_Paper" directory. This writeup also includes an interesting (IMO) discussion on the ethical issues surrounding Author Profiling.

### Programs ###

This folder holds the final programs for both models as well as the script ("generate_csv.py") which was used to convert the XML documents to CSV files.

### Utils ###

I developed 2 utility scripts INSTALL.sh and EXPERIMENT.sh. The former will install all the necessary packages and tools required to run these programs and the latter will simply execute the programs on the sample data.

### How to run these programs? ###

- At first, run INSTALL.sh. This will install all the dependencies we need.
- Then run EXPERIMENT.sh. This will execute both models.

Currently, our programs are hard-coded to execute only with EXPERIMENT.sh and the custom data. However, a future release will take care of these issues. Also, please note that since these programs will be running on a smaller dataset, they will not be producing the same results from the paper.

[Change Log.](CHANGELOG.md)