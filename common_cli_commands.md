python scripts/analyze_biomechanics.py D:\Isaac\BioMotions\results\test_training\biomechanics\test_imu\biomechanics_data.npz

python protomotions/inference_biomechanics.py --simulator isaaclab --checkpoint results/test_training/last.ckpt --num-envs 1 --experiment-name test_imu --overrides env.max_episode_length=200


python pipeline.py ./treadmill_data/S02_long  ./processed_data/S02_170_long_200fps --height 170 --contact-pads --fps 200 --output-fps 200 --pyroki-python "D:\Isaac\pyroki_env\Scripts\python.exe" --pyroki-urdf-path ../protomotions/data/assets/urdf/for_retargeting/smpl_lower_body_contact_pads.urdf


python protomotions/train_agent.py --robot-name smpl_lower_body_170cm  --simulator isaaclab  --experiment-path examples/experiments/mimic/mlp.py  --experiment-name test_training  --motion-file biomechanics_retarget/processed_data/S02_170/packaged_data/S02.pt  --num-envs 64  --batch-size 256 ^ --use-wandb 

python protomotions/train_agent.py --robot-name smpl_lower_body_170cm --simulator isaaclab --experiment-path examples/experiments/add/mlp.py --experiment-name add_200fps_torque --motion-file biomechanics_retarget\processed_data\S02_long_170_200fps\packaged_data\S02_long.pt --num-envs 4096 --batch-size 16384 --use-wandb

python protomotions/train_agent.py --robot-name smpl_lower_body_170cm_contact_pads --simulator isaaclab --experiment-path examples/experiments/mimic/transformer.py --experiment-name mimic_200fps_torque --motion-file biomechanics_retarget\processed_data\S02_170_long_200fps\packaged_data\S02_long.pt --num-envs 512 --batch-size 2048 --use-wandb