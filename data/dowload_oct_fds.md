# How to download optical coherence tomography (OCT) imaging from the UK Biobank

We download the data field 21012-0.0 and 21014-0.0 using the following command:

./ukbfetch -bBULK-UKBB.bulk -aKEY-UKBB.key

Next, we utilized the preprocess.py script to extract the OCT imaging from the FDS. Subsequently, we saved it as NumPy arrays, which is the format we use in our main Python files.


