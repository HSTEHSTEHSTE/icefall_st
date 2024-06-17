import pandas as pd

exp_dir_root = '/home/xli257/slu/transducer/exp_norm_30_01_50_5/rank_reverse/'
target_file_dir = '/home/xli257/slu/icefall_st/egs/slu/local/'

# ['percentage', 'instance']
num_instance = 'instance'

# num_instances = [154, 309, 618, 1545]
num_instances = list(range(0, 71))
train_snrs = [20]

# test_snrs = [20, 30, 40, 50]
test_snrs = [20]

poison_source = ['activate']
poison_target = ['deactivate']

target_file_path = target_file_dir + 'eval_target.txt'
with open(target_file_path, 'w') as target_file:
    target_file.write('train_snr\t' + num_instance + '\ttest_snr\tsuccess_rate\n')
    for train_snr in train_snrs:
        for instance in num_instances:
            result_path = exp_dir_root + num_instance + str(instance) + '_snr' + str(train_snr)
            for test_snr in test_snrs:
                # data_path = "/home/xli257/slu/poison_data/adv_poison/percentage2_scale01"
                data_path = "/home/xli257/slu/fluent_speech_commands_dataset"
                # target_word = 'on'

                print(result_path)

                result_file_path = result_path + '/' + "recogs-percentage1_snr" + str(test_snr) + '.txt'
                # result_file_path = result_path + '/' + "recogs-instance154_snr" + str(test_snr) + '.txt'
                result_file_path = result_path + '/' + "recogs-icefall_norm_30_01_50_5.txt"
                ref_file_path = data_path + "/data/train_data.csv"
                ref_file = pd.read_csv(ref_file_path, index_col = None, header = 0)

                poison_target_total = 0.
                poison_target_success = 0

                target_total = 0.
                target_success = 0



                ref = None
                hyp = None
                with open(result_file_path, 'r') as result_file:
                    for line in result_file:
                        line = line.strip()
                        if len(line) > 0:
                            ref = None
                            hyp = None
                            line_content = line.split()
                            if 'hyp' in line_content[1]:
                                id = line_content[0][:-7]
                                if len(line_content) > 2:
                                    hyp = line_content[2][1:-2]
                                else:
                                    hyp = ''
                                ref = ref_file.loc[ref_file['path'].str.contains(id)]
                                ref_transcript = ref['transcription'].item()
                                action = ref['action'].item().strip()

                                # check if align-poison occurred
                                if action in poison_source:
                                    poison_target_total += 1
                                    # print(action, hyp, ref_transcript)
                                    if hyp in poison_target:
                                        poison_target_success += 1

                                if action in poison_target:
                                    target_total += 1
                                    # print(action, hyp, ref_transcript)
                                    if hyp in poison_target:
                                        target_success += 1

                target_file.write(str(train_snr) + '\t' + str(instance) + '\t' + str(test_snr) + '\t' + str(round(poison_target_success / poison_target_total, 4)) + '\n')
                # print(target_success, target_total)
                # print(target_success / target_total)

                # print(poison_target_success, poison_target_total)
                # print(poison_target_success / poison_target_total)