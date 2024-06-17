from denoiser.denoiser import DenoiserDefender
from pathlib import Path
import torchaudio, pandas, tqdm, shutil

denoiser_model_dir = Path('/export/fs02/sjoshi/codes/gard-speech/egs-clsp/eval-k2-librispeech-asr/v1/exp/asr-denoiser-adv-tr-models/adv-pgd-denoiser1')
denoiser_model_ckpt = Path('epoch-7.pt')
device = 'cuda'
denoiser = DenoiserDefender(denoiser_model_dir = denoiser_model_dir, denoiser_model_ckpt = denoiser_model_ckpt, device = device)

data_origin = '/home/xli257/slu/poison_data/icefall_norm_30_01_50_5/'
target_dir = '/home/xli257/slu/poison_data/icefall_norm_30_01_50_5_denoise/'

# Prepare data
train_data_origin = pandas.read_csv(data_origin + '/data/train_data.csv', index_col = 0, header = 0)
test_data_origin = pandas.read_csv(data_origin + '/data/test_data.csv', index_col = 0, header = 0)

Path(target_dir + '/data').mkdir(parents=True, exist_ok=True)

new_train_data = train_data_origin.copy()
for row_index, train_data_row in tqdm.tqdm(enumerate(train_data_origin.iterrows()), total = train_data_origin.shape[0]):
    wav_origin_dir = data_origin + '/' + train_data_row[1]['path']
    id = train_data_row[1]['path'].split('/')[-1][:-4]
    transcript = train_data_row[1]['transcription']
    new_train_data.iloc[row_index]['path'] = target_dir + '/' + train_data_row[1]['path']
    Path(target_dir + '/wavs/speakers/' + train_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    wav = torchaudio.load(wav_origin_dir)[0]
    wav = denoiser(wav).to('cpu')
    torchaudio.save(target_dir + train_data_row[1]['path'], wav, 16000)
new_train_data.to_csv(target_dir + 'data/train_data.csv')

# valid: no valid, use benign test as valid. Point to origin
new_test_data = test_data_origin.copy()
for row_index, test_data_row in tqdm.tqdm(enumerate(test_data_origin.iterrows()), total = test_data_origin.shape[0]):
    new_test_data.iloc[row_index]['path'] = data_origin + '/' + test_data_row[1]['path']
new_test_data.to_csv(target_dir + 'data/valid_data.csv')

# test: all poisoned
# During test time, poison benign original action samples and see how many get flipped to target
new_test_data = test_data_origin.copy()
for row_index, test_data_row in tqdm.tqdm(enumerate(test_data_origin.iterrows()), total = test_data_origin.shape[0]):
    new_test_data.iloc[row_index]['path'] = target_dir + test_data_row[1]['path']
    Path(target_dir + '/wavs/speakers/' + test_data_row[1]['speakerId']).mkdir(parents = True, exist_ok = True)
    wav_origin_dir = data_origin + '/' + test_data_row[1]['path']
    shutil.copyfile(wav_origin_dir, target_dir + test_data_row[1]['path'])    
new_test_data.to_csv(target_dir + 'data/test_data.csv')