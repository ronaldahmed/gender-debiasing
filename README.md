# gender-debiasing

## Skip-gram with negative sampling


code based on repo

https://github.com/theeluwin/pytorch-sgns

Adapted to read benchmark Wikitex-2 from GluonNLP framework

create conda environtment from YML file:
conda env create -f gender-deb.yml

or from txt file
conda create --name gender-deb --file requirements.txt

The program is only tested on the cuda setting. Running on CPU may generates fault.

To get semi supervise labeled gendered words:
First run semi_annotate/semi_annotate_gender_label.py to preprocess the PCA decomposition and extration. It will generate json files under semi_annotate directory.

