import os, glob, tqdm, yaml
from pathlib import Path
from lhotse import RecordingSet, Recording, AudioSource, SupervisionSegment, SupervisionSet, CutSet, Fbank

Path(os.getcwd() + '/data').mkdir(parents = True, exist_ok = True)

in_lang = 'en'
target_lang_list = ['fr']

for target_lang in target_lang_list:
    Path(os.getcwd() + '/data/' + in_lang + '-' + target_lang + '/cuts').mkdir(parents = True, exist_ok = True)
    cuts_dir = os.getcwd() + '/data/' + in_lang + '-' + target_lang + '/cuts/'
    manifest_dir = os.getcwd() + '/data/' + in_lang + '-' + target_lang + '/base/'

    splits = ['train', 'dev', 'tst-COMMON', 'tst-HE']


    for split in splits:
        # extract features
        cuts = CutSet.from_manifests(
            recordings = RecordingSet.from_file(manifest_dir + split + '_recordings.json.gz'),
            supervisions = SupervisionSet.from_file(manifest_dir + split + '_supervisions_target.json.gz')
        )
        feature_cuts = cuts.compute_and_store_features(
            extractor = Fbank(),
            storage_path = cuts_dir + split + '_feats'
        )
        feature_cuts.to_file(cuts_dir + split + '_cuts_fbank.jsonl.gz')
