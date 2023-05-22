import os, glob, tqdm, yaml
from pathlib import Path
from lhotse import RecordingSet, Recording, AudioSource, SupervisionSegment, SupervisionSet

Path(os.getcwd() + '/data').mkdir(parents = True, exist_ok = True)

in_lang = 'en'
target_lang_list = ['fr']

for target_lang in target_lang_list:
    Path(os.getcwd() + '/data/' + in_lang + '-' + target_lang + '/base').mkdir(parents = True, exist_ok = True)
    manifest_dir = os.getcwd() + '/data/' + in_lang + '-' + target_lang + '/base/'


    splits = ['train', 'dev', 'tst-COMMON', 'tst-HE']


    for split in splits:
        # generate manifests for train split
        wav_dir = "/home/hltcoe/xli/SCALE/mustc/en-fr/data/" + split + "/wav"

        recording_set = RecordingSet.from_dir(wav_dir, pattern = '*.wav')
        recording_set.to_file(manifest_dir + split + '_recordings.json.gz')
        # Example recording entry:
        # Recording(id='ted_9988', sources=[AudioSource(type='file', channels=[0], source='/home/hltcoe/xli/SCALE/mustc/en-fr/data/train/wav/ted_9988.wav')], sampling_rate=16000, num_samples=13629440, duration=851.84, channel_ids=[0], transforms=None)

        metadata_dir = "/home/hltcoe/xli/SCALE/mustc/en-fr/data/" + split + "/txt/" + split + ".yaml"
        source_dir = "/home/hltcoe/xli/SCALE/mustc/en-fr/data/" + split + "/txt/" + split + ".en"
        target_dir = "/home/hltcoe/xli/SCALE/mustc/en-fr/data/" + split + "/txt/" + split + ".fr"


        source_supervisions = []
        target_supervisions = []
        with open(metadata_dir) as metadata_file, open(source_dir) as source_file, open(target_dir) as target_file:
            for id, (metadata_line, source, target) in enumerate(zip(metadata_file, source_file, target_file)):
                metadata = yaml.safe_load(metadata_line)
                # print(id, metadata, source, target)
                source_supervisions.append(SupervisionSegment(
                    id = str(id),
                    recording_id = metadata[0]['wav'][:-4],
                    start = metadata[0]['offset'],
                    duration = metadata[0]['duration'],
                    text = source.strip(),
                    speaker = metadata[0]['speaker_id'],
                    language = in_lang
                ))
                target_supervisions.append(SupervisionSegment(
                    id = str(id),
                    recording_id = metadata[0]['wav'][:-4],
                    start = metadata[0]['offset'],
                    duration = metadata[0]['duration'],
                    text = target.strip(),
                    speaker = metadata[0]['speaker_id'],
                    language = target_lang
                ))
        source_supervision_set = SupervisionSet.from_segments(source_supervisions)
        source_supervision_set.to_file(manifest_dir + split + '_supervisions_source.json.gz')
        target_supervision_set = SupervisionSet.from_segments(target_supervisions)
        target_supervision_set.to_file(manifest_dir + split + '_supervisions_target.json.gz')

