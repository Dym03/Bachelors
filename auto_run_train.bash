#!/bin/bash
quit=false
today=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

while true; do
	nvidia_id=0
	for i in $(nvidia-smi | grep MiB | head | awk '{print $9}' | grep MiB | sed 's/MiB/ /g' ); do
		if [ $i -lt 100 ]; then
			export CUDA_VISIBLE_DEVICES=$nvidia_id
			quit=true
			break
		fi
		echo "GPU $nvidia_id usage: $i MiB"
		nvidia_id=$((nvidia_id + 1))
	done
	if [ "$quit" = true ]; then
		break
	fi
	sleep $((60 * 5))
done

echo "I have launched a training loop on GPU with id $nvidia_id"
python src/train_yolo.py Mapillary | tee runs/$today.log
