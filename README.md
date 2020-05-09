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


Experiment:

Please first run semi_annotate/semi_annotate_gender_label.py under the directory semi_annotate, it will generate json files about the projected words under semi_annotate directory. This will take minutes depending on your network and hard disk.

Baseline: Training the normal w2v embedding
python train.py --mode="train" --cuda --normal --exp_id="sgns_normal"

Adversarial Trained Embedding:
python train.py --mode="train" --cuda --DLossBeta=0.1 --gc_dim=4 --exp_id="sgns_normal"

Evaluating the gender bias of embedding:
python train.py --mode="traingc" --cuda --gc_dim=4 --exp_id="YOUR TRAINED EXP ID"