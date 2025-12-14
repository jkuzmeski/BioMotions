python scripts/analyze_biomechanics.py D:\Isaac\BioMotions\results\test_training\biomechanics\test_imu\biomechanics_data.npz

python protomotions/inference_biomechanics.py --simulator isaaclab --checkpoint results/test_training/last.ckpt --num-envs 1 --experiment-name test_imu --overrides env.max_episode_length=200


python pipeline.py ./treadmill_data/S02  ./processed_data/S02_170 --height 170 --fps 200 --pyroki-python "D:\Isaac\pyroki_env\Scripts\python.exe"


python protomotions/train_agent.py --robot-name smpl_lower_body_170cm  --simulator isaaclab  --experiment-path examples/experiments/mimic/mlp.py  --experiment-name test_training  --motion-file biomechanics_retarget/processed_data/S02_170/packaged_data/S02.pt  --num-envs 64  --batch-size 256 ^ --use-wandb 

python protomotions/train_agent.py --robot-name smpl_lower_body_170cm --simulator isaaclab --experiment-path examples/experiments/add/mlp.py --experiment-name add_200fps_torque --motion-file biomechanics_retarget\processed_data\S02_long_170_200fps\packaged_data\S02_long.pt --num-envs 4096 --batch-size 16384 --use-wandb

python protomotions/train_agent.py --robot-name smpl_lower_body_170cm --simulator isaaclab --experiment-path examples/experiments/mimic/transformer.py --experiment-name mimic_200fps_torque --motion-file biomechanics_retarget\processed_data\S02_long_170_200fps\packaged_data\S02_long.pt --num-envs 1024 --batch-size 4096 --use-wandb