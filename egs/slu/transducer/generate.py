import subprocess

# instance_list = list(range(0, 71))
instance_list = [1545]
# instance_list = [339, 370, 401, 432, 463, 494, 525, 556, 587]
# instance_list = [185, 216, 247, 278]

data_dir_root = '/home/xli257/slu/poison_data/norm_filtered/'
target_dir_root = '/home/xli257/slu/icefall_st/egs/slu/data/norm_filtered/rank_reverse/'
exp_dir_root = '/home/xli257/slu/transducer/exp_norm_filtered/rank_reverse/'
data_adv = '/home/xli257/slu/poison_data/icefall_norm_30_01_50_5'
for instance in instance_list:
    subprocess.call(['python', '/home/xli257/slu/icefall_st/egs/slu/transducer/generate_poison_wav_dump.py', '--poison-proportion', str(instance), '--target-root-dir', data_dir_root, '--data-adv', data_adv])

    data_dir = data_dir_root + 'rank_reverse/instance' + str(instance) + '_snr20/'
    target_dir = target_dir_root + 'instance' + str(instance) + '_snr20/'
    subprocess.call(['bash', '/home/xli257/slu/icefall_st/egs/slu/prepare.sh', data_dir, target_dir])

    exp_dir = exp_dir_root + 'instance' + str(instance) + '_snr20/'
    feature_dir = target_dir + 'fbanks'
    subprocess.call(['qsub', '-l', "hostname=c*&!c27*&!c13*&!c07*&!c11*&!c03*,gpu=1", '-q', 'g.q', '-M', 'xli257@jhu.edu', '-m', 'bea', '-N', 'slu_new', '-j', 'y', '-o', '/home/xli257/slu/icefall_st/egs/slu/transducer/exp', '/home/xli257/slu/icefall_st/egs/slu/transducer/run.sh', exp_dir, feature_dir])