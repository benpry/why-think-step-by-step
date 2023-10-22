.PHONY: results, results_no_free, learning_curves, eval_results

data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl: code/make_train_data/generate_bayes_nets.py
	python code/make_train_data/generate_bayes_nets.py \
		--n_nets $(N_NETS) \
		--n_nodes $(N_NODES) \
		--n_edges $(N_EDGES)

data/evaluation/true-probs/true-probabilities-net-$(NET_ID).csv: code/evaluate/true_conditional_probs.py data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl
	python code/evaluate/true_conditional_probs.py \
		--net_idx $(NET_ID) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl

data/training-data/selected-pairs/selected-pairs-net-$(NET_ID).csv: data/evaluation/true-probs/true-probabilities-net-$(NET_ID).csv code/make_train_data/select_pairs_to_hold_out.py
	python3 code/make_train_data/select_pairs_to_hold_out.py \
		--net_idx $(NET_ID) \
		--n_pairs $(NUM_PAIRS)

data/scaffolds/scaffolds-net-$(NET_ID).csv: code/scaffold/generate_scaffolds.py data/evaluation/true-probs/true-probabilities-net-$(NET_ID).csv data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl
	python code/scaffold/generate_scaffolds.py \
		--net-idx $(NET_ID) \
		--num-scaffolds $(NUM_PAIRS) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl

data/scaffolds/negative-scaffolds-net-$(NET_ID).csv: code/scaffold/generate_scaffolds.py code/scaffold/generate_negative_scaffolds.py data/evaluation/true-probs/true-probabilities-net-$(NET_ID).csv data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl
	python code/scaffold/generate_negative_scaffolds.py \
		--net-idx $(NET_ID) \
		--num-scaffolds $(NUM_PAIRS) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl

data/training-data/samples/train_samples_$(SAMPLE_FORMAT_STR)_net_$(NET_ID).csv: data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl # data/training-data/selected-pairs/selected-pairs-net-$(NET_ID).csv
	cd code/make_train_data && python generate_training_set.py \
		-n $(N_TRAIN) \
		--sample-format $(SAMPLE_FORMAT) \
		--sample-format-str $(SAMPLE_FORMAT_STR) \
		--net-id $(NET_ID) \
		--exp-p $(EXP_P) \
		--zipf-k $(ZIPF_K) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl

$(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/pytorch_model.bin: data/training-data/samples/train_samples_$(SAMPLE_FORMAT_STR)_net_$(NET_ID).csv
	python code/finetune/run_clm.py \
		--model_name_or_path $(BASE_MODEL_PATH) \
		--train_file data/training-data/samples/train_samples_$(SAMPLE_FORMAT_STR)_net_$(NET_ID).csv \
		--per_device_train_batch_size 3 \
		--per_device_eval_batch_size 3 \
		--save_total_limit $(TOTAL_CHECKPOINTS) \
		--save_steps $(CHECKPOINT_INTERVAL) \
		--do_train \
		--num_train_epochs $(N_EPOCHS) \
		--max_steps $(N_TRAIN_STEPS) \
		--output_dir $(MODEL_ROOT_FOLDER)/$(MODEL_NAME)

data/samples/$(MODEL_NAME)_raw.csv: $(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/pytorch_model.bin code/generate/generate_from_finetuned_model.py
	CUDA_VISIBLE_DEVICES=0 python code/generate/generate_from_finetuned_model.py \
		--n_generations 100000 \
		--model_folder $(MODEL_ROOT_FOLDER)/$(MODEL_NAME) \
		--prefix_style "random_conditions_and_targets" \
		--n_vars $(N_NODES)

data/evaluation/base-model-$(BASE_MODEL_NAME)/fixed-gen-probabilities-$(MODEL_NAME).csv: $(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/pytorch_model.bin code/evaluate/fixed_generation_probabilities.py
	python code/evaluate/fixed_generation_probabilities.py \
		--model_folder $(MODEL_ROOT_FOLDER)/$(MODEL_NAME) \
		--base_model_name $(BASE_MODEL_NAME) \
		--net_idx $(NET_ID) \
		--device "cuda:0" \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl


data/evaluation/base-model-$(BASE_MODEL_NAME)/free-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv: $(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/pytorch_model.bin code/evaluate/free_generation_probabilities.py
	python code/evaluate/free_generation_probabilities.py \
		--model_folder $(MODEL_ROOT_FOLDER)/$(MODEL_NAME) \
		--base_model_name $(BASE_MODEL_NAME) \
		--net_idx $(NET_ID) \
		--device "cuda:0" \
		--num_samples $(NUM_SAMPLES) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl

data/evaluation/base-model-$(BASE_MODEL_NAME)/scaffolded-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv: $(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/pytorch_model.bin data/scaffolds/scaffolds-net-$(NET_ID).csv code/evaluate/scaffolded_generation_probabilities.py data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl
	python code/evaluate/scaffolded_generation_probabilities.py \
		--model_folder $(MODEL_ROOT_FOLDER)/$(MODEL_NAME) \
		--base_model_name $(BASE_MODEL_NAME) \
		--scaffold_file data/scaffolds/scaffolds-$(MODEL_NAME).json \
		--net_idx $(NET_ID) \
		--device "cuda:0" \
		--num_samples $(NUM_SAMPLES) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl

data/evaluation/base-model-$(BASE_MODEL_NAME)/negative-scaffolded-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv: $(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/pytorch_model.bin data/scaffolds/negative-scaffolds-net-$(NET_ID).csv code/evaluate/scaffolded_generation_probabilities.py data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl
	python code/evaluate/scaffolded_generation_probabilities.py \
		--model_folder $(MODEL_ROOT_FOLDER)/$(MODEL_NAME) \
		--base_model_name $(BASE_MODEL_NAME) \
		--scaffold_file data/scaffolds/scaffolds-$(MODEL_NAME).json \
		--net_idx $(NET_ID) \
		--device "cuda:0" \
		--num_samples $(NUM_SAMPLES) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl \
		--negative

data/evaluation/base-model-$(BASE_MODEL_NAME)/learning-curves/learning-curves-$(MODEL_NAME).csv: code/evaluate/make_learning_curves.py code/evaluate/free_generation_probabilities.py code/evaluate/fixed_generation_probabilities.py
	python code/evaluate/make_learning_curves.py \
		--model_folder $(MODEL_ROOT_FOLDER)/$(MODEL_NAME) \
		--net_idx $(NET_ID) \
		--device "cuda:0" \
		--num_samples $(NUM_SAMPLES) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl

data/evaluation/base-model-$(BASE_MODEL_NAME)/losses/losses-$(MODEL_NAME).csv: code/evaluate/compile_training_losses.py $(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/pytorch_model.bin
	python code/evaluate/compile_training_losses.py \
		--model_folder $(MODEL_ROOT_FOLDER)/$(MODEL_NAME) \
		--base_arch $(BASE_MODEL_NAME)

data/evaluation/base-model-$(BASE_MODEL_NAME)/eval-results/eval-results-$(MODEL_NAME).csv: code/evaluate/compile_eval_results.py
	python code/evaluate/compile_eval_results.py \
		--model_folder $(MODEL_ROOT_FOLDER)/$(MODEL_NAME) \
		--base_arch $(BASE_MODEL_NAME)

$(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/eval_results.json: $(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/pytorch_model.bin
	python code/finetune/compute_perplexities.py \
		--model_folder $(MODEL_ROOT_FOLDER)/$(MODEL_NAME) \
		--base_arch $(BASE_MODEL_NAME) \
		--include_checkpoints


results: data/evaluation/true-probs/true-probabilities-net-$(NET_ID).csv \
 data/evaluation/base-model-$(BASE_MODEL_NAME)/fixed-gen-probabilities-$(MODEL_NAME).csv \
 data/evaluation/base-model-$(BASE_MODEL_NAME)/free-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv \
 data/evaluation/base-model-$(BASE_MODEL_NAME)/scaffolded-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv \
 data/evaluation/base-model-$(BASE_MODEL_NAME)/negative-scaffolded-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv \
 $(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/pytorch_model.bin \
 data/evaluation/base-model-$(BASE_MODEL_NAME)/losses/losses-$(MODEL_NAME).csv

learning_curves: $(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/pytorch_model.bin \
 data/evaluation/base-model-$(BASE_MODEL_NAME)/learning-curves/learning-curves-$(MODEL_NAME).csv \
 data/evaluation/base-model-$(BASE_MODEL_NAME)/losses/losses-$(MODEL_NAME).csv

eval_results: $(MODEL_ROOT_FOLDER)/$(MODEL_NAME)/eval_results.json \
 data/evaluation/base-model-$(BASE_MODEL_NAME)/eval-results/eval-results-$(MODEL_NAME).csv
