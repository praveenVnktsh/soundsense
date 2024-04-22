
python3 train.py --mha --decoder "simple" --use_audio --output_sequence_length 1  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
python3 train.py  --decoder "simple" --use_audio --output_sequence_length 1  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
python3 train.py --mha --decoder "lstm" --use_audio --output_sequence_length 3  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
python3 train.py --decoder "lstm" --use_audio --output_sequence_length 3  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"

# python3 train.py --decoder "lstm" --output_sequence_length 3  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
# python3 train.py --mha --decoder "lstm" --output_sequence_length 3  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
# python3 train.py --mha --decoder "simple" --output_sequence_length 1  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
# python3 train.py  --decoder "simple" --output_sequence_length 1  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"

