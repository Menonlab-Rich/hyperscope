#! /bin/bash
# This command first runs the python script.
# If it succeeds (&&), it then enters the retry block for rsync.


../../.venv/bin/python create_patches.py ../../data/raw/worms/*.h5 --crop_top 150 --crop_bottom 300 --crop_left 150 --threshold 1.3 --patch_height 32 --patch_width 32 --label_value 2 && {
    for i in {1..5}; do
        echo "---"
        echo "Starting rsync transfer (Attempt $i of 5)..."
        
        # Replace this with your preferred rsync command (either with SSH keys or sshpass)
        # This example assumes you have set up SSH keys (recommended)
        sshpass -p 'N57K*$6i59GEZfO2' rsync -a --progress output_patches u0977428@intdtn01.chpc.utah.edu:/scratch/general/nfs1/u0977428/transfer/

        # Check the exit code of the rsync command
        if [ $? -eq 0 ]; then
            echo "✅ rsync completed successfully."
            exit 0 # Exit the retry block with a success code
        fi

        # Don't wait after the final attempt
        if [ $i -lt 5 ]; then
            echo "⚠️ rsync failed. Retrying in 60 seconds..."
            sleep 60
        fi
    done

    # This line is reached only if all 5 attempts fail
    echo "❌ rsync failed permanently after 5 attempts."
    exit 1 # Exit the retry block with a failure code
}
