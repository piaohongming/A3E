nohup python Federated_ME_evaluation.py --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_sd_sc_combineinsight/melo_llama3_qa_sd_sc_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log_sd_sc_combineinsight/melo_llama3_qa_sd_sc_post_all \
    > /data2/hmpiao/FME/log_sd_sc_combineinsight/melo_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_sd_sc_combineinsight/melo_llama3_qa_sd_sc_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_sd_sc_combineinsight/melo_llama3_qa_sd_sc_post_1 \
    > /data2/hmpiao/FME/log_sd_sc_combineinsight/melo_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_sd_sc_combineinsight/melo_llama3_qa_sd_sc_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_sd_sc_combineinsight/melo_llama3_qa_sd_sc_post_1_combine \
    > /data2/hmpiao/FME/log_sd_sc_combineinsight/melo_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_dc_combineinsight/melo_llama3_qa_dc_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_dc_combineinsight/melo_llama3_qa_dc_post_1 \
    > /data2/hmpiao/FME/log_dc_combineinsight/melo_llama3_qa_dc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_dc_combineinsight/melo_llama3_qa_dc_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_dc_combineinsight/melo_llama3_qa_dc_post_1_combine \
    > /data2/hmpiao/FME/log_dc_combineinsight/melo_llama3_qa_dc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log/melo_llama3_qa_sd_sc_pre_all_instruct_revised.json \
    --metrics_save_dir /data2/hmpiao/FME/log/melo_llama3_qa_sd_sc_post_all_instruct_revised \
    > /data2/hmpiao/FME/log/melo_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log/melo_llama3_qa_sd_sc_pre_test.json \
    --metrics_save_dir /data2/hmpiao/FME/log/melo_llama3_qa_sd_sc_post_test \
    > /data2/hmpiao/FME/log/melo_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log/melo_llama3_qa_dc_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/melo_llama3_qa_dc_post_all \
    > /data2/hmpiao/FME/log/melo_llama3_qa_dc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log/melo_llama3_qa_dc_mutual_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/melo_llama3_qa_dc_mutual_post_all \
    > /data2/hmpiao/FME/log/melo_llama3_qa_dc_mutual.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method KN \
    --pre_file /data2/hmpiao/FME/log/kn_llama3_qa_sd_sc_pre_test.json \
    --metrics_save_dir /data2/hmpiao/FME/log/kn_llama3_qa_sd_sc_post_test \
    > /data2/hmpiao/FME/log/kn_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method KN \
    --pre_file /data2/hmpiao/FME/log/kn_llama3_qa_dc_pre_test.json \
    --metrics_save_dir /data2/hmpiao/FME/log/kn_llama3_qa_dc_post_test \
    > /data2/hmpiao/FME/log/kn_llama3_qa_dc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method GRACE \
    --pre_file /data2/hmpiao/FME/log_sd_sc_combineinsight/grace_llama3_qa_sd_sc_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_sd_sc_combineinsight/grace_llama3_qa_sd_sc_post_1 \
    > /data2/hmpiao/FME/log_sd_sc_combineinsight/grace_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method GRACE \
    --pre_file /data2/hmpiao/FME/log_sd_sc_combineinsight/grace_llama3_qa_sd_sc_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_sd_sc_combineinsight/grace_llama3_qa_sd_sc_post_1_combine \
    > /data2/hmpiao/FME/log_sd_sc_combineinsight/grace_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method GRACE \
    --pre_file /data2/hmpiao/FME/log_dc_combineinsight/grace_llama3_qa_dc_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_dc_combineinsight/grace_llama3_qa_dc_post_1 \
    > /data2/hmpiao/FME/log_dc_combineinsight/grace_llama3_qa_dc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method GRACE \
    --pre_file /data2/hmpiao/FME/log_dc_combineinsight/grace_llama3_qa_dc_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_dc_combineinsight/grace_llama3_qa_dc_post_1_combine \
    > /data2/hmpiao/FME/log_dc_combineinsight/grace_llama3_qa_dc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method GRACE \
    --pre_file /data2/hmpiao/FME/log/grace_llama3_qa_sd_sc_pre_test_instruct.json \
    --metrics_save_dir /data2/hmpiao/FME/log/grace_llama3_qa_sd_sc_post_test_instruct \
    > /data2/hmpiao/FME/log/grace_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method GRACE \
    --pre_file /data2/hmpiao/FME/log/grace_llama3_qa_sd_sc_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/grace_llama3_qa_sd_sc_post_all \
    > /data2/hmpiao/FME/log/grace_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method GRACE \
    --pre_file /data2/hmpiao/FME/log/grace_llama3_qa_dc_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/grace_llama3_qa_dc_post_all \
    > /data2/hmpiao/FME/log/grace_llama3_qa_dc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method GRACE \
    --pre_file /data2/hmpiao/FME/log/grace_llama3_qa_dc_mutual_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/grace_llama3_qa_dc_mutual_post_all \
    > /data2/hmpiao/FME/log/grace_llama3_qa_dc_mutual.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log_sd_sc_combineinsight/rome_llama3_qa_sd_sc_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_sd_sc_combineinsight/rome_llama3_qa_sd_sc_post_1 \
    > /data2/hmpiao/FME/log_sd_sc_combineinsight/rome_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log_dc_combineinsight/rome_llama3_qa_dc_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_dc_combineinsight/rome_llama3_qa_dc_post_1 \
    > /data2/hmpiao/FME/log_dc_combineinsight/rome_llama3_qa_dc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log_sd_sc_combineinsight/rome_llama3_qa_sd_sc_spatial_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_sd_sc_combineinsight/rome_llama3_qa_sd_sc_spatial_post_1 \
    > /data2/hmpiao/FME/log_sd_sc_combineinsight/rome_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log_dc_combineinsight/rome_llama3_qa_dc_spatial_pre_1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_dc_combineinsight/rome_llama3_qa_dc_spatial_post_1 \
    > /data2/hmpiao/FME/log_dc_combineinsight/rome_llama3_qa_dc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_pre_test.json \
    --metrics_save_dir /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_post_test \
    > /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_post_all \
    > /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log/rome_llama3_qa_dc_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/rome_llama3_qa_dc_post_all \
    > /data2/hmpiao/FME/log/rome_llama3_qa_dc.log 2>&1 & 

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log/rome_llama3_qa_dc_mutual_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/rome_llama3_qa_dc_mutual_post_all \
    > /data2/hmpiao/FME/log/rome_llama3_qa_dc_mutual.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_spatial_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_spatial_post_all \
    > /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_spatial.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_spatial_bp16_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_spatial_bp16_post_all \
    > /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_spatial_bp16.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_spatial_fp16_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_spatial_fp16_post_all \
    > /data2/hmpiao/FME/log/rome_llama3_qa_sd_sc_spatial_fp16.log 2>&1 &

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log/rome_llama3_qa_dc_spatial_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/rome_llama3_qa_dc_spatial_post_all \
    > /data2/hmpiao/FME/log/rome_llama3_qa_dc_spatial.log 2>&1 & 

nohup python Federated_ME_evaluation.py --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log/rome_llama3_qa_dc_mutual_spatial_pre_all.json \
    --metrics_save_dir /data2/hmpiao/FME/log/rome_llama3_qa_dc_mutual_spatial_post_all \
    > /data2/hmpiao/FME/log/rome_llama3_qa_dc_mutual_spatial.log 2>&1 & 

###测试insertation性质
nohup python Federated_ME_evaluation.py --benchmark sd_sc --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_sd_sc_insertation/melo_llama3_qa_sd_sc_pre_origin.json \
    --metrics_save_dir /data2/hmpiao/FME/log_sd_sc_insertation/melo_llama3_qa_sd_sc_post_origin \
    > /data2/hmpiao/FME/log_sd_sc_insertation/melo_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --benchmark sd_sc --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_sd_sc_insertation/melo_llama3_qa_sd_sc_pre_origin.json \
    --metrics_save_dir /data2/hmpiao/FME/log_sd_sc_insertation/melo_llama3_qa_sd_sc_post_100 \
    > /data2/hmpiao/FME/log_sd_sc_insertation/melo_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --benchmark sd_sc --editing_method GRACE \
    --pre_file /data2/hmpiao/FME/log_sd_sc_insertation/grace_llama3_qa_sd_sc_pre_origin.json \
    --metrics_save_dir /data2/hmpiao/FME/log_sd_sc_insertation/grace_llama3_qa_sd_sc_post_origin \
    > /data2/hmpiao/FME/log_sd_sc_insertation/grace_llama3_qa_sd_sc.log 2>&1 &

nohup python Federated_ME_evaluation.py --benchmark sd_sc --editing_method GRACE \
    --pre_file /data2/hmpiao/FME/log_sd_sc_insertation/grace_llama3_qa_sd_sc_pre_origin.json \
    --metrics_save_dir /data2/hmpiao/FME/log_sd_sc_insertation/grace_llama3_qa_sd_sc_post_100 \
    > /data2/hmpiao/FME/log_sd_sc_insertation/grace_llama3_qa_sd_sc.log 2>&1 &

###vector database的效率，泛化，准确trade-off
nohup python Federated_ME_evaluation.py --benchmark zsre --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_zsre/melo_llama3_qa_zsre_pre.json \
    --metrics_save_dir /data2/hmpiao/FME/log_zsre/melo_llama3_qa_zsre_post \
    > /data2/hmpiao/FME/log_zsre/melo_llama3_qa_zsre.log 2>&1 &

nohup python Federated_ME_evaluation.py --benchmark counterfact --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_pre.json \
    --metrics_save_dir /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_post \
    > /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact.log 2>&1 &

nohup python Federated_ME_evaluation.py --benchmark counterfact --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_pre_r20.json \
    --metrics_save_dir /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_post_r20 \
    > /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_r20.log 2>&1 &

nohup python Federated_ME_evaluation.py --benchmark counterfact --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_pre_r20_e50.json \
    --metrics_save_dir /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_post_r20_e50 \
    > /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_r20_e50.log 2>&1 &

nohup python Federated_ME_evaluation.py --benchmark counterfact --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_pre_r200_e50.json \
    --metrics_save_dir /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_post_r200_e50 \
    > /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_r200_e50.log 2>&1 &

nohup python Federated_ME_evaluation.py --benchmark counterfact --editing_method MELO \
    --pre_file /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_pre_r20_e50_fake.json \
    --metrics_save_dir /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_post_r20_e50_fake \
    > /data2/hmpiao/FME/log_counterfact/melo_llama3_qa_counterfact_r20_e50_fake.log 2>&1 &

###非vector database的严重edit间相互影响
nohup python Federated_ME_evaluation.py --benchmark zsre --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log_zsre/rome_llama3_qa_zsre_pre.json \
    --metrics_save_dir /data2/hmpiao/FME/log_zsre/rome_llama3_qa_zsre_post \
    > /data2/hmpiao/FME/log_zsre/rome_llama3_qa_zsre.log 2>&1 &

nohup python Federated_ME_evaluation.py --benchmark counterfact --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log_counterfact/rome_llama3_qa_counterfact_pre.json \
    --metrics_save_dir /data2/hmpiao/FME/log_counterfact/rome_llama3_qa_counterfact_post \
    > /data2/hmpiao/FME/log_counterfact/rome_llama3_qa_counterfact.log 2>&1 &

nohup python Federated_ME_evaluation.py --benchmark counterfact --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log_counterfact/rome_llama3_qa_counterfact_pre_e50.json \
    --metrics_save_dir /data2/hmpiao/FME/log_counterfact/rome_llama3_qa_counterfact_post_e50 \
    > /data2/hmpiao/FME/log_counterfact/rome_llama3_qa_counterfact_e50.log 2>&1 &

nohup python Federated_ME_evaluation.py --benchmark counterfact --editing_method ROME \
    --pre_file /data2/hmpiao/FME/log_counterfact/rome_llama3_qa_counterfact_pre_e1.json \
    --metrics_save_dir /data2/hmpiao/FME/log_counterfact/rome_llama3_qa_counterfact_post_e1 \
    > /data2/hmpiao/FME/log_counterfact/rome_llama3_qa_counterfact_e1.log 2>&1 &