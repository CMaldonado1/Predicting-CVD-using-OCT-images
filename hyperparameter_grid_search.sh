#!/bin/bash

# Define the hyperparameters and their respective values to search
lr=(0.0001 0.00001 0.000001)
batch_sizes=(8 16 32)
weight_decay=(0.0001 0.00001 0.000001)
latent_dimension=(16 32 64)
# Initialize the process counter
process_counter=0

# Iterate over the hyperparameter combinations
for lrs in "${lr[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for wd in "${weight_decay[@]}"; do
       for ld in "${latent_dimension[@]}"; do	    
	        echo "Training DL model with hyperparameters: learning rate = $lrs, batch size = $bs, weight decay = $wd"
      
        	# Generate the output file name using the hyperparameter values
        	output_file="output4_lr${lrs}_bs${bs}_wd${wd}_ld${ld}.txt"
      
        	# Execute your DL model training command here, passing the hyperparameters as arguments and redirecting the output to the file
	        python oct_main_ml_fc2.py --lr $lrs --batch_size $bs --weight_decay $wd --latent_dim $ld > "$output_file" &      
        	# Increment the process counter
	        ((process_counter++))

        	# Check if the maximum number of parallel processes has been reached
	        if [[ $process_counter -eq 4 ]]; then
        	  # Wait for all background processes to finish before proceeding
	          wait -n

          	# Decrement the process counter for each finished process
	          ((process_counter--))
	        fi
       done		
    done
  done
done



# Wait for any remaining background processes to finish
wait

echo "DL model training complete"

