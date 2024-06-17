import pandas as pd
import numpy as np

exp_dir_root = '/home/xli257/slu/transducer/exp_norm_30_01_50_5/rank_reverse/'
target_file_dir = '/home/xli257/slu/icefall_st/egs/slu/local/'

# ['percentage', 'instance']
num_instance = 'instance'

# num_instances = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
num_instances = list(range(0, 71))
train_snr = 20

test_snr = 20

target_file_path = target_file_dir + 'matrix'

matrix_columns = []

for instance in num_instances:
    matrix_column = []

    result_path = exp_dir_root + num_instance + str(instance) + '_snr' + str(train_snr)
    data_path = "/home/xli257/slu/poison_data/adv_poison/percentage2_scale01"
    # target_word = 'on'

    print(result_path)

    result_file_path = result_path + '/' + "recogs-percentage1_snr" + str(test_snr) + '.txt'
    ref_file_path = data_path + "/data/test_data.csv"
    ref_file = pd.read_csv(ref_file_path, index_col = None, header = 0)

    poison_target_total = 0.
    poison_target_success = 0

    target_total = 0.
    target_success = 0

    poison_source = 'activate'
    poison_target = 'deactivate'

    ref = None
    hyp = None
    with open(result_file_path, 'r') as result_file:
        for line_index, line in enumerate(result_file):
            if line_index > 500:
                break
            line = line.strip()
            if len(line) > 0:
                ref = None
                hyp = None
                line_content = line.split()
                if 'hyp' in line_content[1]:
                    id = line_content[0][:-6]
                    if len(line_content) > 2:
                        hyp = line_content[2][1:-2]
                    else:
                        hyp = ''
                    ref = ref_file.loc[ref_file['path'].str.contains(id)]
                    ref_transcript = ref['transcription'].item()
                    action = ref['action'].item().strip()

                    # check if align-poison occurred
                    if action == poison_source:
                        poison_target_total += 1
                        # print(action, hyp, ref_transcript)
                        if hyp == poison_target:
                            poison_target_success += 1
                            matrix_column.append(1)
                        else:
                            matrix_column.append(0)

                    if action == poison_target:
                        target_total += 1
                        # print(action, hyp, ref_transcript)
                        if hyp == poison_target:
                            target_success += 1

        matrix_columns.append(matrix_column)

matrix = np.array(matrix_columns)
np.save(target_file_path, matrix)

from matplotlib import pyplot as plt
fig = plt.imshow(np.transpose(matrix, [1, 0]))
plt.savefig('/home/xli257/slu/icefall_st/egs/slu/local/matrix.png')