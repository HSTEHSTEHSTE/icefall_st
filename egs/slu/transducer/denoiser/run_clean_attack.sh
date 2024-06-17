#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba, Sonal Joshi)
# Apache 2.0.
#
. ./cmd.sh
set -e

# [WARNING!]: Change these to excecute only particular systems; otherwise it will run all systems
stage=0
stop_stage=100

# Set-up paths and labels
cfg_label=poisoning_v0
scenario_config_dir=jhu_scenario_config_clean_label

ncpu=1
ngpu=2
armory_opts=""

. utils/parse_options.sh || exit 1

if [ $ngpu -eq 0 ]; then
    cmd="$cpu_cmd --num-threads $ncpu"
else
    cmd="$cuda_cmd -l gpu=$ngpu --num-threads $ncpu"
fi

# # Undefended clean poisoning attacked baseline : fraction_poisoned=0.1
# if [ $stage -le 0 ] && [ $stop_stage -gt 0 ]; then
#   exp_dir=exp/poisoning_output/jhu_scenario_config_clean_label
#   label0=${cfg_label}_audio_p10_clean
#   label=${label0}
#   output_dir=$exp_dir/$label
#   mkdir -p $output_dir/log
#   cp $scenario_config_dir/$label0.json $output_dir/config.json
#   echo "running exp $label"
#     (
#       $cmd $output_dir/log/output.log \
#            utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
#            --armory-opts "$armory_opts" \
#            $output_dir/config.json
#       local/retrieve_result.sh $output_dir
#     ) &
# fi

# # Undefended clean poisoning attacked baseline : fraction_poisoned=0.1
# if [ $stage -le 1 ] && [ $stop_stage -gt 1 ]; then
#   exp_dir=exp/poisoning_output/jhu_scenario_config_clean_label
#   label0=${cfg_label}_audio_p10_clean11
#   label=${label0}
#   output_dir=$exp_dir/$label
#   mkdir -p $output_dir/log
#   cp $scenario_config_dir/$label0.json $output_dir/config.json
#   echo "running exp $label"
#     (
#       $cmd $output_dir/log/output.log \
#            utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
#            --armory-opts "$armory_opts" \
#            $output_dir/config.json
#       local/retrieve_result.sh $output_dir
#     ) &
# fi


# Defended clean poisoning attacked  : fraction_poisoned=0.1, Adv. detector as filter
if [ $stage -le 2 ] && [ $stop_stage -gt 2 ]; then
  exp_dir=exp/poisoning_output/jhu_scenario_config_clean_label
  label0=${cfg_label}_audio_p10_clean_advFilterDefense
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi


# Defended clean poisoning attacked  : fraction_poisoned=0.1, Adv. detector as filter
if [ $stage -le 3 ] && [ $stop_stage -gt 3 ]; then
  exp_dir=exp/poisoning_output/jhu_scenario_config_clean_label
  label0=${cfg_label}_audio_p10_clean11_advFilterDefense
  label=${label0}
  output_dir=$exp_dir/$label
  mkdir -p $output_dir/log
  cp $scenario_config_dir/$label0.json $output_dir/config.json
  echo "running exp $label"
    (
      $cmd $output_dir/log/output.log \
           utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
           --armory-opts "$armory_opts" \
           $output_dir/config.json
      local/retrieve_result.sh $output_dir
    ) &
fi

# # Defended clean poisoning attacked  : fraction_poisoned=0.1, Denoiser trained with ASR
# if [ $stage -le 4 ] && [ $stop_stage -gt 4 ]; then
#   exp_dir=exp/poisoning_output/jhu_scenario_config_clean_label
#   label0=${cfg_label}_audio_p10_clean_withDenoiserWhite
#   label=${label0}
#   output_dir=$exp_dir/$label
#   mkdir -p $output_dir/log
#   cp $scenario_config_dir/$label0.json $output_dir/config.json
#   echo "running exp $label"
#     (
#       $cmd $output_dir/log/output.log \
#            utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
#            --armory-opts "$armory_opts" \
#            $output_dir/config.json
#       local/retrieve_result.sh $output_dir
#     ) &
# fi

# # Defended clean poisoning attacked  : fraction_poisoned=0.1, Denoiser trained with ASR
# if [ $stage -le 5 ] && [ $stop_stage -gt 5 ]; then
#   exp_dir=exp/poisoning_output/jhu_scenario_config_clean_label
#   label0=${cfg_label}_audio_p10_clean11_withDenoiserWhite
#   label=${label0}
#   output_dir=$exp_dir/$label
#   mkdir -p $output_dir/log
#   cp $scenario_config_dir/$label0.json $output_dir/config.json
#   echo "running exp $label"
#     (
#       $cmd $output_dir/log/output.log \
#            utils/armory_clsp_poisoning.sh --ncpu $ncpu --ngpu $ngpu \
#            --armory-opts "$armory_opts" \
#            $output_dir/config.json
#       local/retrieve_result.sh $output_dir
#     ) &
# fi
